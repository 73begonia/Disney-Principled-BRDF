#define numSamplesPerFrame 32
#define maxPathLength 4
#define PI 3.1415926

//*******************************************************************************//
//DXR

RaytracingAccelerationStructure scene : register(t0, space100);
RWBuffer<float4> tracerOutBuffer : register(u0);

struct Vertex
{
    float3 position;
    float3 normal;
    float2 texCoord;
};

struct Material
{
    float3 emissive;
    float3 baseColor;
    float subsurface;
    float metallic;
    float specular;
    float specularTint;
    float roughness;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    float IOR;
    float transmission;
    
    bool bLight;
};

struct GPUSceneObject
{
    uint vertexOffset;
    uint tridexOffset;
    uint materialIdx;

    row_major float4x4 modelMatrix;
};

StructuredBuffer<GPUSceneObject> objectBuffer : register(t0);
StructuredBuffer<Vertex> vertexBuffer : register(t1);
Buffer<uint3> tridexBuffer : register(t2);
StructuredBuffer<Material> materialBuffer : register(t3);

Texture2D<float4> g_hdrMap : register(t4);
Texture2D<float4> g_cacheHdr : register(t5);
SamplerState g_sampler : register(s0);

cbuffer GLOBAL_CONSTANTS : register(b0)
{
    float3 cameraPos;
    uint accumulatedFrames;
    float4x4 invViewProj;
    int hdrResolution;
}

cbuffer OBJECT_CONSTANTS : register(b1)
{
    uint objIdx;
}

struct RayPayload
{
    float3 radiance;
    float3 attenuation;
    float3 hitPos;
    float3 bounceDir;
    uint rayDepth;
    uint seed;
    
    float3 f_r;
    float pdf_brdf;
    float NdotL;
};

RayDesc Ray(in float3 origin, in float3 direction, in float tMin, in float tMax)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    return ray;
}

void computeNormal(out float3 normal, in BuiltInTriangleIntersectionAttributes attr)
{
    GPUSceneObject obj = objectBuffer[objIdx];

    uint3 tridex = tridexBuffer[obj.tridexOffset + PrimitiveIndex()];
    Vertex vtx0 = vertexBuffer[obj.vertexOffset + tridex.x];
    Vertex vtx1 = vertexBuffer[obj.vertexOffset + tridex.y];
    Vertex vtx2 = vertexBuffer[obj.vertexOffset + tridex.z];

    float t0 = 1.0f - attr.barycentrics.x - attr.barycentrics.y;
    float t1 = attr.barycentrics.x;
    float t2 = attr.barycentrics.y;

    float3x3 transform = (float3x3) obj.modelMatrix;

    normal = normalize(mul(transform, t0 * vtx0.normal + t1 * vtx1.normal + t2 * vtx2.normal));
}

//*******************************************************************************//
//Utility Function

float2 sampleSphericalMap(float3 v)
{
    float2 uv = float2(atan2(v.z, v.x), asin(v.y));
    uv /= float2(2.0 * PI, PI);
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

float3 hdrColor(float3 v)
{
    float2 uv = sampleSphericalMap(normalize(v));
    float3 color = g_hdrMap.SampleLevel(g_sampler, uv, 0).rgb;
    return color;
}

//// 采样预计算的 HDR cache
//float3 sampleHdr(float xi_1, float xi_2)
//{
//    float2 xy = g_cacheHdr.SampleLevel(g_sampler, float2(xi_1, xi_2), 0).rg;
//    xy.y = 1.0 - xy.y;

//    // 获取角度
//    float phi = 2.0 * PI * (xy.x - 0.5); // [-pi ~ pi]
//    float theta = PI * (xy.y - 0.5); // [-pi/2 ~ pi/2]   

//    // 球坐标计算方向
//    float3 L = float3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi));

//    return L;
//}

//// 输入光线方向 L 获取 HDR 在该位置的概率密度
//// hdr 分辨率为 4096 x 2048 --> hdrResolution = 4096
//float hdrPdf(float3 L, int hdrResolution)
//{
//    float2 uv = sampleSphericalMap(normalize(L)); // 方向向量转 uv 纹理坐标

//    float pdf = g_cacheHdr.SampleLevel(g_sampler, uv, 0).b;
    
//    float theta = PI * (0.5 - uv.y); // theta 范围 [-pi/2 ~ pi/2]
//    float sin_theta = max(sin(theta), 1e-10);

//    // 球坐标和图片积分域的转换系数
//    float p_convert = float(hdrResolution * hdrResolution / 2) / (2.0 * PI * PI * sin_theta);
    
//    return pdf * p_convert;
//}

//float misMixWeight(float a, float b)
//{
//    float t = a * a;
//    return t / (b * b + t);
//}

uint getNewSeed(uint param1, uint param2, uint numPermutation)
{
    uint s0 = 0;
    uint v0 = param1;
    uint v1 = param2;

    for (uint perm = 0; perm < numPermutation; perm++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

uint wang_hash(inout uint seed)
{
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

float rand(inout uint seed)
{
    return float(wang_hash(seed)) / 4294967296.0;
}

float2 cranleyPattersonRotation(float2 p, inout uint pseed)
{
    float u = float(wang_hash(pseed)) / 4294967296.0;
    float v = float(wang_hash(pseed)) / 4294967296.0;

    p.x += u;
    if (p.x > 1)
        p.x -= 1;
    if (p.x < 0)
        p.x += 1;

    p.y += v;
    if (p.y > 1)
        p.y -= 1;
    if (p.y < 0)
        p.y += 1;

    return p;
}

//*******************************************************************************//
//BRDF`s and PDF`s

float3 sampleHemisphere(inout uint seed)
{
    float z = rand(seed);
    float r = max(0, sqrt(1.0 - z * z));
    float phi = 2.0 * PI * rand(seed);
    return float3(r * cos(phi), r * sin(phi), z);
}

float3 toNormalHemisphere(float3 v, float3 N)
{
    float3 helper = float3(1, 0, 0);
    if (abs(N.x) > 0.999)
        helper = float3(0, 0, 1);
    float3 tangent = normalize(cross(N, helper));
    float3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

float schlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m;
}

float GTR1(float NdotH, float a)
{
    if (a >= 1)
        return 1 / PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

float3 brdf_Evaluate(float3 V, float3 N, float3 L, in Material material)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0 || NdotV <= 0)
        return 0.f;
    
    float3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);
    
    //各种颜色
    float3 Cdlin = material.baseColor;
    float Cdlum = 0.3 * Cdlin.r + 0.6 * Cdlin.g + 0.1 * Cdlin.b;
    float3 Ctint = (Cdlum > 0) ? (Cdlin / Cdlum) : 1;
    float3 Cspec = material.specular * lerp(1, Ctint, material.specularTint);
    float3 Cspec0 = lerp(0.08 * Cspec, Cdlin, material.metallic); // 0° 镜面反射颜色
    float3 Csheen = lerp(1, Ctint, material.sheenTint); // 织物颜色
    
    //漫反射
    float Fd90 = 0.5 + 2.0 * LdotH * LdotH * material.roughness;
    float FL = schlickFresnel(NdotL);
    float FV = schlickFresnel(NdotV);
    float Fd = lerp(1.0, Fd90, FL) * lerp(1.0, Fd90, FV);
    
    //次表面散射
    float Fss90 = LdotH * LdotH * material.roughness;
    float Fss = lerp(1.0, Fss90, FL) * lerp(1.0, Fss90, FV);
    float ss = 1.25 * (Fss * (1.0 / (NdotL + NdotV) - 0.5) + 0.5);
    
    // 镜面反射
    float alpha = material.roughness * material.roughness;
    float Ds = GTR2(NdotH, alpha);
    float FH = schlickFresnel(LdotH);
    float3 Fs = lerp(Cspec0, 1, FH);
    float Gs = smithG_GGX(NdotL, material.roughness);
    Gs *= smithG_GGX(NdotV, material.roughness);
    
    //清漆
    float Dr = GTR1(NdotH, lerp(0.1, 0.001, material.clearcoatGloss));
    float Fr = lerp(0.04, 1.0, FH);
    float Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25);
    
    // sheen
    float3 Fsheen = FH * material.sheen * Csheen;
    
    float3 diffuse = (1.0 / PI) * lerp(Fd, ss, material.subsurface) * Cdlin + Fsheen;
    float3 specular = Gs * Fs * Ds;
    float3 clearcoat = 0.25 * Gr * Fr * Dr * material.clearcoat;
    
    return diffuse * (1.0 - material.metallic) + specular + clearcoat;
}

void getTangent(float3 N, inout float3 tangent, inout float3 bitangent)
{
    float3 helper = float3(1, 0, 0);
    if (abs(N.x) > 0.999)
        helper = float3(0, 0, 1);
    bitangent = normalize(cross(N, helper));
    tangent = normalize(cross(N, bitangent));
}

float sqr(float x)
{
    return x * x;
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1 / (PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
}

float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
}

float3 brdf_Evaluate_aniso(float3 V, float3 N, float3 L, float3 X, float3 Y, in Material material)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0 || NdotV <= 0)
        return 0.f;
    
    float3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);
    
    //各种颜色
    float3 Cdlin = material.baseColor;
    float Cdlum = 0.3 * Cdlin.r + 0.6 * Cdlin.g + 0.1 * Cdlin.b;
    float3 Ctint = (Cdlum > 0) ? (Cdlin / Cdlum) : 1;
    float3 Cspec = material.specular * lerp(1, Ctint, material.specularTint);
    float3 Cspec0 = lerp(0.08 * Cspec, Cdlin, material.metallic); // 0° 镜面反射颜色
    float3 Csheen = lerp(1, Ctint, material.sheenTint); // 织物颜色
    
    //漫反射
    float Fd90 = 0.5 + 2.0 * LdotH * LdotH * material.roughness;
    float FL = schlickFresnel(NdotL);
    float FV = schlickFresnel(NdotV);
    float Fd = lerp(1.0, Fd90, FL) * lerp(1.0, Fd90, FV);
    
    //次表面散射
    float Fss90 = LdotH * LdotH * material.roughness;
    float Fss = lerp(1.0, Fss90, FL) * lerp(1.0, Fss90, FV);
    float ss = 1.25 * (Fss * (1.0 / (NdotL + NdotV) - 0.5) + 0.5);
    
    // 镜面反射 -- 各向异性
    float aspect = sqrt(1.0 - material.anisotropic * 0.9);
    float ax = max(0.001, sqr(material.roughness) / aspect);
    float ay = max(0.001, sqr(material.roughness) * aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = schlickFresnel(LdotH);
    float3 Fs = lerp(Cspec0, 1, FH);
    float Gs;
    Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    
    //清漆
    float Dr = GTR1(NdotH, lerp(0.1, 0.001, material.clearcoatGloss));
    float Fr = lerp(0.04, 1.0, FH);
    float Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25);
    
    // sheen
    float3 Fsheen = FH * material.sheen * Csheen;
    
    float3 diffuse = (1.0 / PI) * lerp(Fd, ss, material.subsurface) * Cdlin + Fsheen;
    float3 specular = Gs * Fs * Ds;
    float3 clearcoat = 0.25 * Gr * Fr * Dr * material.clearcoat;
    
    return diffuse * (1.0 - material.metallic) + specular + clearcoat;
}

//*******************************************************************************//
//Importance Sampling

// 1 ~ 8 维的 sobol 生成矩阵
static const uint V[8 * 32] =
{
    2147483648, 1073741824, 536870912, 268435456, 134217728, 67108864, 33554432, 16777216, 8388608, 4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1,
    2147483648, 3221225472, 2684354560, 4026531840, 2281701376, 3422552064, 2852126720, 4278190080, 2155872256, 3233808384, 2694840320, 4042260480, 2290614272, 3435921408, 2863267840, 4294901760, 2147516416, 3221274624, 2684395520, 4026593280, 2281736192, 3422604288, 2852170240, 4278255360, 2155905152, 3233857728, 2694881440, 4042322160, 2290649224, 3435973836, 2863311530, 4294967295,
    2147483648, 3221225472, 1610612736, 2415919104, 3892314112, 1543503872, 2382364672, 3305111552, 1753219072, 2629828608, 3999268864, 1435500544, 2154299392, 3231449088, 1626210304, 2421489664, 3900735488, 1556135936, 2388680704, 3314585600, 1751705600, 2627492864, 4008611328, 1431684352, 2147543168, 3221249216, 1610649184, 2415969680, 3892340840, 1543543964, 2382425838, 3305133397,
    2147483648, 3221225472, 536870912, 1342177280, 4160749568, 1946157056, 2717908992, 2466250752, 3632267264, 624951296, 1507852288, 3872391168, 2013790208, 3020685312, 2181169152, 3271884800, 546275328, 1363623936, 4226424832, 1977167872, 2693105664, 2437829632, 3689389568, 635137280, 1484783744, 3846176960, 2044723232, 3067084880, 2148008184, 3222012020, 537002146, 1342505107,
    2147483648, 1073741824, 536870912, 2952790016, 4160749568, 3690987520, 2046820352, 2634022912, 1518338048, 801112064, 2707423232, 4038066176, 3666345984, 1875116032, 2170683392, 1085997056, 579305472, 3016343552, 4217741312, 3719483392, 2013407232, 2617981952, 1510979072, 755882752, 2726789248, 4090085440, 3680870432, 1840435376, 2147625208, 1074478300, 537900666, 2953698205,
    2147483648, 1073741824, 1610612736, 805306368, 2818572288, 335544320, 2113929216, 3472883712, 2290089984, 3829399552, 3059744768, 1127219200, 3089629184, 4199809024, 3567124480, 1891565568, 394297344, 3988799488, 920674304, 4193267712, 2950604800, 3977188352, 3250028032, 129093376, 2231568512, 2963678272, 4281226848, 432124720, 803643432, 1633613396, 2672665246, 3170194367,
    2147483648, 3221225472, 2684354560, 3489660928, 1476395008, 2483027968, 1040187392, 3808428032, 3196059648, 599785472, 505413632, 4077912064, 1182269440, 1736704000, 2017853440, 2221342720, 3329785856, 2810494976, 3628507136, 1416089600, 2658719744, 864310272, 3863387648, 3076993792, 553150080, 272922560, 4167467040, 1148698640, 1719673080, 2009075780, 2149644390, 3222291575,
    2147483648, 1073741824, 2684354560, 1342177280, 2281701376, 1946157056, 436207616, 2566914048, 2625634304, 3208642560, 2720006144, 2098200576, 111673344, 2354315264, 3464626176, 4027383808, 2886631424, 3770826752, 1691164672, 3357462528, 1993345024, 3752330240, 873073152, 2870150400, 1700563072, 87021376, 1097028000, 1222351248, 1560027592, 2977959924, 23268898, 437609937
};

// 格林码 
uint grayCode(uint i)
{
    return i ^ (i >> 1);
}

// 生成第 d 维度的第 i 个 sobol 数
float sobol(uint d, uint i)
{
    uint result = 0;
    uint offset = d * 32;
    for (uint j = 0; i; i >>= 1, j++) 
        if (i & 1)
            result ^= V[j + offset];

    return float(result) * (1.0f / float(0xFFFFFFFFU));
}

// 生成第 i 帧的第 b 次反弹需要的二维随机向量
float2 sobolFloat2(uint i, uint b)
{
    float u = sobol(b * 2, grayCode(i));
    float v = sobol(b * 2 + 1, grayCode(i));
    return float2(u, v);
}

float3 sampleHemisphere(float xi_1, float xi_2)
{
    float z = xi_1;
    float r = max(0, sqrt(1.0 - z * z));
    float phi = 2.0 * PI * xi_2;
    return float3(r * cos(phi), r * sin(phi), z);
}

// 余弦加权的法向半球采样
float3 sampleCosineHemisphere(float xi_1, float xi_2, float3 N)
{
    // 均匀采样 xy 圆盘然后投影到 z 半球
    float r = sqrt(xi_1);
    float theta = xi_2 * 2.0 * PI;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - x * x - y * y);

    // 从 z 半球投影到法向半球
    float3 L = toNormalHemisphere(float3(x, y, z), N);
    return L;
}

// GTR2 重要性采样
float3 sampleGTR2(float xi_1, float xi_2, float3 V, float3 N, float alpha)
{
    
    float phi_h = 2.0 * PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0 - xi_2) / (1.0 + (alpha * alpha - 1.0) * xi_2));
    float sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));

    // 采样 "微平面" 的法向量 作为镜面反射的半角向量 h 
    float3 H = float3(sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h);
    H = toNormalHemisphere(H, N); // 投影到真正的法向半球

    // 根据 "微法线" 计算反射光方向
    float3 L = reflect(-V, H);

    return L;
}

// GTR1 重要性采样
float3 sampleGTR1(float xi_1, float xi_2, float3 V, float3 N, float alpha)
{
    
    float phi_h = 2.0 * PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0 - pow(alpha * alpha, 1.0 - xi_2)) / (1.0 - alpha * alpha));
    float sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));

    // 采样 "微平面" 的法向量 作为镜面反射的半角向量 h 
    float3 H = float3(sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h);
    H = toNormalHemisphere(H, N); // 投影到真正的法向半球

    // 根据 "微法线" 计算反射光方向
    float3 L = reflect(-V, H);

    return L;
}

// 按照辐射度分布分别采样三种 BRDF
float3 sampleBRDF(float xi_1, float xi_2, float xi_3, float3 V, float3 N, in Material material)
{
    float alpha_GTR1 = lerp(0.1, 0.001, material.clearcoatGloss);
    float alpha_GTR2 = max(0.001, sqr(material.roughness));
    
    // 辐射度统计
    float r_diffuse = (1.0 - material.metallic);
    float r_specular = 1.0;
    float r_clearcoat = 0.25 * material.clearcoat;
    float r_sum = r_diffuse + r_specular + r_clearcoat;

    // 根据辐射度计算概率
    float p_diffuse = r_diffuse / r_sum;
    float p_specular = r_specular / r_sum;
    float p_clearcoat = r_clearcoat / r_sum;

    // 按照概率采样
    float rd = xi_3;

    // 漫反射
    if (rd <= p_diffuse)
    {
        return sampleCosineHemisphere(xi_1, xi_2, N);
    }
    // 镜面反射
    else if (p_diffuse < rd && rd <= p_diffuse + p_specular)
    {
        return sampleGTR2(xi_1, xi_2, V, N, alpha_GTR2);
    }
    // 清漆
    else if (p_diffuse + p_specular < rd)
    {
        return sampleGTR1(xi_1, xi_2, V, N, alpha_GTR1);
    }
    return float3(0, 1, 0);
}

// 获取 BRDF 在 L 方向上的概率密度
float brdf_Pdf(float3 V, float3 N, float3 L, in Material material)
{
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0 || NdotV <= 0)
        return 0;

    float3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);
     
    // 镜面反射 -- 各向同性
    float alpha = max(0.001, sqr(material.roughness));
    float Ds = GTR2(NdotH, alpha);
    float Dr = GTR1(NdotH, lerp(0.1, 0.001, material.clearcoatGloss)); // 清漆

    // 分别计算三种 BRDF 的概率密度
    float pdf_diffuse = NdotL / PI;
    float pdf_specular = Ds * NdotH / (4.0 * dot(L, H));
    float pdf_clearcoat = Dr * NdotH / (4.0 * dot(L, H));

    // 辐射度统计
    float r_diffuse = (1.0 - material.metallic);
    float r_specular = 1.0;
    float r_clearcoat = 0.25 * material.clearcoat;
    float r_sum = r_diffuse + r_specular + r_clearcoat;

    // 根据辐射度计算选择某种采样方式的概率
    float p_diffuse = r_diffuse / r_sum;
    float p_specular = r_specular / r_sum;
    float p_clearcoat = r_clearcoat / r_sum;

    // 根据概率混合 pdf
    float pdf = p_diffuse * pdf_diffuse
              + p_specular * pdf_specular
              + p_clearcoat * pdf_clearcoat;

    pdf = max(1e-7, pdf);
    return pdf;
}