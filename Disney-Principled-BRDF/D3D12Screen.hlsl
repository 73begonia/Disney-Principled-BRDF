struct VSInput
{
    uint vertexId : SV_VertexID;
};

typedef
struct VSOutput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD;
} PSInput;

VSOutput VSMain(VSInput input)
{
    const float4 screen[4] =
    {
        float4(-1, -1, 0, 1),
		float4(-1, 1, 0, 1),
		float4(1, -1, 0, 1),
		float4(1, 1, 0, 1)
    };
    const float2 uv[4] =
    {
        float2(0, 1),
		float2(0, 0),
		float2(1, 1),
		float2(1, 0)
    };

    VSOutput result;
    result.position = screen[input.vertexId];
    result.uv = uv[input.vertexId];

    return result;
}

Texture2D g_texture : register(t0);
SamplerState g_sampler : register(s0);

static const float exposure = 0.45;

float4 PSMain(PSInput input) : SV_TARGET
{
    float3 hdrColor = abs(g_texture.Sample(g_sampler, input.uv).rgb);

    float3 ldrColor = 1.0 - exp(-exposure * hdrColor);
    
    //float3 ldrColor = hdrColor / (hdrColor + 1.0);

    float4 outColor = float4(pow(ldrColor, 1 / 2.2), 1);

    return outColor;
}