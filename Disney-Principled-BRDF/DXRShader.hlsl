//#pragma pack_matrix( row_major )    // It does not work!
#include "sampling.hlsli"

float3 tracePath(in float3 startPos, in float3 startDir, inout uint seed)
{
    float3 radiance = 0.0f;
    float3 attenuation = 1.0f;

    RayDesc ray = Ray(startPos, startDir, 1e-4f, 1e27f);
    RayPayload prd;
    prd.seed = seed;
    prd.rayDepth = 0;

    while (prd.rayDepth < maxPathLength)
    {
        TraceRay(scene, 0, ~0, 0, 1, 0, ray, prd);

        radiance += attenuation * prd.radiance;
        attenuation *= prd.attenuation;

        ray.Origin = prd.hitPos;
        ray.Direction = prd.bounceDir;
        ++prd.rayDepth;
    }

    return radiance;
}

[shader("raygeneration")]
void rayGen()
{
    float2 launchIdx = DispatchRaysIndex().xy;
    float2 launchDim = DispatchRaysDimensions().xy;
    uint bufferOffset = launchDim.x * launchIdx.y + launchIdx.x;

    uint seed = getNewSeed(bufferOffset, accumulatedFrames, 8);
    
    float3 newRadiance = 0.0f;
    float3 avrRadiance = 0.0f;

    RayPayload payload;

    for (uint i = 0; i < numSamplesPerFrame; i++)
    {
        float2 uv = ((launchIdx + float2(rand(seed) - 0.5, rand(seed) - 0.5)) / launchDim) * 2.f - 1.f;
        uv.y = -uv.y;
        
        RayDesc ray;
        ray.Origin = cameraPos;
        
        float4 world = mul(float4(uv, 1.0f, 1.0f), invViewProj);
        ray.Direction = normalize(world).xyz;

        newRadiance += tracePath(cameraPos, (world).xyz, seed);
    }

    newRadiance *= 1.0f / float(numSamplesPerFrame);
    
    if (accumulatedFrames == 0)
        avrRadiance = newRadiance;
    else
        avrRadiance = lerp(tracerOutBuffer[bufferOffset].xyz, newRadiance, 1.f / (accumulatedFrames + 1.0f));

    tracerOutBuffer[bufferOffset] = float4(avrRadiance, 1.0f);
}

[shader("closesthit")]
void closestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    float3 hitNormal = 0.f;
    computeNormal(hitNormal, attribs);

    GPUSceneObject obj = objectBuffer[objIdx];
    uint mtlIdx = obj.materialIdx;
    Material material = materialBuffer[mtlIdx];
    
    float3 radiance = 0.f;
    float3 attenuation = 1.f;
    
    float3 V = -WorldRayDirection();
    float3 N = hitNormal;
    
    //普通版本
    //float2 xy = sobolFloat2(accumulatedFrames, payload.rayDepth);
    //xy = cranleyPattersonRotation(xy, payload.seed);
    //float3 L = sampleHemisphere(xy.x, xy.y);
    //L = toNormalHemisphere(L, N);
    
    //float pdf = 1.0 / (2.0 * PI);
    //float cosine_o = max(0, dot(V, N));
    //float cosine_i = max(0, dot(L, N));
    
    //float3 tangent, bitangent;
    //getTangent(N, tangent, bitangent);
    //float3 f_r = brdf_Evaluate_aniso(V, N, L, tangent, bitangent, material);
    
    //float3 Le = material.emissive;
    //radiance = Le * f_r * cosine_i / pdf;
    //attenuation = f_r * cosine_i / pdf;
    
    //{
        //float3 ro = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
        //float3 rd = sampleHdr(rand(payload.seed), rand(payload.seed));
    
        //RayDesc hdrTestRay = Ray(ro, rd, 1e-4f, 1e27f);
    
        //float3 L = rd;
        //float pdf_light = hdrPdf(L, hdrResolution);
        //float3 f_r = brdf_Evaluate(-WorldRayDirection(), N, L, material);
        //float pdf_brdf = brdf_Pdf(-WorldRayDirection(), N, L, material);
        
        //RayPayload prdHdr;
        //prdHdr.radiance = 0;
        //prdHdr.f_r = f_r;
        //prdHdr.pdf_brdf = pdf_brdf;
        //prdHdr.NdotL = dot(N, L);
        
        //TraceRay(scene, 0, ~0, 0, 1, 1, hdrTestRay, prdHdr);
    
        //radiance += prdHdr.radiance;
    //}
    
    //重要性采样版本
    float2 xy = sobolFloat2(accumulatedFrames, payload.rayDepth);
    xy = cranleyPattersonRotation(xy, payload.seed);
    float xi_1 = xy.x;
    float xi_2 = xy.y;
    float xi_3 = rand(payload.seed);
    
    float3 L = sampleBRDF(xi_1, xi_2, xi_3, V, N, material);
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0.0 || NdotV <= 0.0)
    {
        payload.radiance = 0;
        payload.attenuation = 1;
        payload.rayDepth = maxPathLength;
        return;
    }
    
    float3 f_r = brdf_Evaluate(V, N, L, material);
    float pdf_brdf = brdf_Pdf(V, N, L, material);
    if (pdf_brdf <= 1e-7)
    {
        payload.radiance = 0;
        payload.attenuation = 1;
        payload.rayDepth = maxPathLength;
        return;
    }
    
    float3 Le = material.emissive;
    radiance += Le * f_r * NdotL / pdf_brdf;
    attenuation = f_r * NdotL / pdf_brdf;
    
    payload.radiance = radiance;
    payload.attenuation = attenuation;
    payload.hitPos = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    payload.bounceDir = L;
    
    payload.f_r = f_r;
    payload.pdf_brdf = pdf_brdf;
    payload.NdotL = NdotL;
}

[shader("miss")]
void missRay(inout RayPayload payload)
{
    //float3 L = WorldRayDirection();
    //float3 color = hdrColor(normalize(L));
    //float pdf_light = hdrPdf(L, hdrResolution);
    
    //float mis_weight = misMixWeight(payload.pdf_brdf, pdf_light);
    //payload.radiance = mis_weight * color * payload.f_r * payload.NdotL / payload.pdf_brdf;
    payload.radiance = hdrColor(normalize(WorldRayDirection()));
    payload.attenuation = 1.0f;
    payload.rayDepth = maxPathLength;
}

[shader("miss")]
void missShadow(inout RayPayload hdrPayload)
{
    //float3 L = WorldRayDirection();
    //float3 color = hdrColor(normalize(L));
    //float pdf_light = hdrPdf(L, hdrResolution);
    
    //float mis_weight = misMixWeight(pdf_light, hdrPayload.pdf_brdf);
    //hdrPayload.radiance = mis_weight * color * hdrPayload.f_r * hdrPayload.NdotL / pdf_light;
    
    //payload.radiance = color * f_r * dot(N, L) / pdf_light;
}
