#include "DXRPathTracer.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <GRSTex/GRS_Texture_Loader.h>

namespace DescriptorID
{
	enum
	{
		outUAV = 0,

		sceneObjectBuff = 1,
		vertexBuff = 2,
		tridexBuff = 3,
		materialBuff = 4,
		hdrBuffer = 5,
		cacheBuffer = 6,

		albedo = 7,
		normal = 8,
		metallic = 9,
		roughness = 10,
		ao = 11,

		maxDescriptors = 32
	};
}

namespace RootParamID
{
	enum
	{
		tableForOutBuffer = 0,
		pointerForAccelerationStructure = 1,
		tableForGeometryInputs = 2,
		pointerForGlobalConstants = 3,
		tableForHDRMap = 4,
		tableForSampler = 5,
		//tableForIronTexture = 6,
		numParams
	};
}

namespace HitGroupParamID
{
	enum
	{
		constantForObject = 0,
		numParams
	};
}

void DXRPathTracer::onMouseDown(WPARAM btnState, int x, int y)
{
	mLastMousePos.x = x;
	mLastMousePos.y = y;

	SetCapture(mTargetWindow);
}

void DXRPathTracer::onMouseUp(WPARAM btnState, int x, int y)
{
	ReleaseCapture();
}

void DXRPathTracer::onMouseMove(WPARAM btnState, int x, int y)
{
	if ((btnState & MK_LBUTTON) != 0)
	{
		float dx = XMConvertToRadians(0.25f * static_cast<float>(x - mLastMousePos.x));
		float dy = XMConvertToRadians(0.25f * static_cast<float>(y - mLastMousePos.y));

		mCamera.pitch(dy);
		mCamera.rotateY(dx);
	}

	mLastMousePos.x = x;
	mLastMousePos.y = y;
}

DXRPathTracer::~DXRPathTracer()
{
	SAFE_RELEASE(hdrTexture);
	SAFE_RELEASE(hdrUploadBuffer);
	SAFE_RELEASE(cacheTexture);
	SAFE_RELEASE(cacheUploadBuffer);
}

DXRPathTracer::DXRPathTracer(HWND hwnd, UINT width, UINT height) :
	mTargetWindow(hwnd), mTracerOutW(width), mTracerOutH(height)
{
	initD3D12();

	createSrvUavHeap();

	loadHdrTexture();

	onSizeChanged(mTracerOutW, mTracerOutH);

	declareRootSignatures();

	buildRaytracingPipeline();

	initializeApplication();
}

void DXRPathTracer::initD3D12()
{
	ThrowIfFailed(createDX12Device(getRTXAdapter().Get())->QueryInterface(IID_PPV_ARGS(&mDevice_v5)));

	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	cqDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(mDevice_v5->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&mCmdQueue_v0)));
	ThrowIfFailed(mDevice_v5->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCmdAllocator_v0)));
	ThrowIfFailed(mDevice_v5->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCmdAllocator_v0.Get(), nullptr, IID_PPV_ARGS(&mCmdList_v4)));

	ThrowIfFailed(mCmdList_v4->Close());
	ThrowIfFailed(mCmdAllocator_v0->Reset());
	ThrowIfFailed(mCmdList_v4->Reset(mCmdAllocator_v0.Get(), nullptr));
	mFence_v0.create(mDevice_v5.Get());
}

void DXRPathTracer::createSrvUavHeap()
{
	D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
	{
		heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		heapDesc.NumDescriptors = DescriptorID::maxDescriptors;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	}

	mDevice_v5->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mSrvUavHeap));
	mSrvDescriptorSize = mDevice_v5->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

float* DXRPathTracer::calculateHdrCache(float* HDR, int width, int height)
{
	float lumSum = 0.0;

	// 初始化 h 行 w 列的概率密度 pdf 并 统计总亮度
	std::vector<std::vector<float>> pdf(height);
	for (auto& line : pdf) line.resize(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float R = HDR[3 * (i * width + j)];
			float G = HDR[3 * (i * width + j) + 1];
			float B = HDR[3 * (i * width + j) + 2];
			float lum = float(0.2 * R + 0.7 * G + 0.1 * B);
			pdf[i][j] = lum;
			lumSum += lum;
		}
	}

	// 概率密度归一化
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			pdf[i][j] /= lumSum;

	// 累加每一列得到 x 的边缘概率密度
	std::vector<float> pdf_x_margin;
	pdf_x_margin.resize(width);
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			pdf_x_margin[j] += pdf[i][j];

	// 计算 x 的边缘分布函数
	std::vector<float> cdf_x_margin = pdf_x_margin;
	for (int i = 1; i < width; i++)
		cdf_x_margin[i] += cdf_x_margin[i - 1];

	// 计算 y 在 X=x 下的条件概率密度函数
	std::vector<std::vector<float>> pdf_y_condiciton = pdf;
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			pdf_y_condiciton[i][j] /= pdf_x_margin[j];

	// 计算 y 在 X=x 下的条件概率分布函数
	std::vector<std::vector<float>> cdf_y_condiciton = pdf_y_condiciton;
	for (int j = 0; j < width; j++)
		for (int i = 1; i < height; i++)
			cdf_y_condiciton[i][j] += cdf_y_condiciton[i - 1][j];

	// cdf_y_condiciton 转置为按列存储
	// cdf_y_condiciton[i] 表示 y 在 X=i 下的条件概率分布函数
	std::vector<std::vector<float>> temp = cdf_y_condiciton;
	cdf_y_condiciton = std::vector<std::vector<float>>(width);
	for (auto& line : cdf_y_condiciton) line.resize(height);
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			cdf_y_condiciton[j][i] = temp[i][j];

	// 穷举 xi_1, xi_2 预计算样本 xy
	// sample_x[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 x
	// sample_y[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 y
	// sample_p[i][j] 表示取 (i, j) 点时的概率密度
	std::vector<std::vector<float>> sample_x(height);
	for (auto& line : sample_x) line.resize(width);
	std::vector<std::vector<float>> sample_y(height);
	for (auto& line : sample_y) line.resize(width);
	std::vector<std::vector<float>> sample_p(height);
	for (auto& line : sample_p) line.resize(width);
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			float xi_1 = float(i) / height;
			float xi_2 = float(j) / width;

			// 用 xi_1 在 cdf_x_margin 中 lower bound 得到样本 x
			int x = int(std::lower_bound(cdf_x_margin.begin(), cdf_x_margin.end(), xi_1) - cdf_x_margin.begin());
			// 用 xi_2 在 X=x 的情况下得到样本 y
			int y = int(std::lower_bound(cdf_y_condiciton[x].begin(), cdf_y_condiciton[x].end(), xi_2) - cdf_y_condiciton[x].begin());

			// 存储纹理坐标 xy 和 xy 位置对应的概率密度
			sample_x[i][j] = float(x) / width;
			sample_y[i][j] = float(y) / height;
			sample_p[i][j] = pdf[i][j];
		}
	}

	// 整合结果到纹理
	// R,G 通道存储样本 (x,y) 而 B 通道存储 pdf(i, j)
	float* cache = new float[width * height * 3];
	//for (int i = 0; i < width * height * 3; i++) cache[i] = 0.0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cache[3 * (i * width + j)] = sample_x[i][j];        // R
			cache[3 * (i * width + j) + 1] = sample_y[i][j];    // G
			cache[3 * (i * width + j) + 2] = sample_p[i][j];    // B
		}
	}

	return cache;
}

void DXRPathTracer::loadHdrTexture()
{
	FLOAT* hdrData = nullptr;
	FLOAT* hdrCachedata = nullptr;
	//int width = 0;
	int height = 0;
	int components = 0;
	DXGI_FORMAT hdrFormat;

	const char* fileName = "../__textures/hdr/peppermint_powerplant_4k.hdr";

	hdrData = stbi_loadf(
	fileName, &mHdrResolution, &height, &components, 0);

	if (hdrData == nullptr)
	{
		throw Error("Failed to load HDR texture");
		return;
	}

	if (components == 3)
	{
		hdrFormat = DXGI_FORMAT_R32G32B32_FLOAT;
	}
	else
	{
		throw Error("Unsupported HDR texture format");
	}

	UINT bpp = (UINT)components * sizeof(float);
	UINT picRowPitch = _align(mHdrResolution * bpp, 8u);

	ThrowIfFalse(LoadTextureFromMem(mCmdList_v4.Get(),
		(BYTE*)hdrData,
		(size_t)mHdrResolution * bpp * height,
		hdrFormat,
		mHdrResolution,
		height,
		picRowPitch,
		hdrUploadBuffer,
		hdrTexture));

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = hdrFormat;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = hdrTexture->GetDesc().MipLevels;
	srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

	D3D12_CPU_DESCRIPTOR_HANDLE hdrTextureHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
	hdrTextureHandle.ptr += (UINT)DescriptorID::hdrBuffer * mSrvDescriptorSize;
	mDevice_v5->CreateShaderResourceView(hdrTexture, &srvDesc, hdrTextureHandle);

	/*hdrCachedata = calculateHdrCache(hdrData, mHdrResolution, height);
	ThrowIfFalse(LoadTextureFromMem(mCmdList_v4.Get(),
		(BYTE*)hdrCachedata,
		(size_t)mHdrResolution * bpp * height,
		hdrFormat,
		mHdrResolution,
		height,
		picRowPitch,
		cacheUploadBuffer,
		cacheTexture));

	D3D12_SHADER_RESOURCE_VIEW_DESC cacheSrvDesc = {};
	cacheSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	cacheSrvDesc.Format = hdrFormat;
	cacheSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	cacheSrvDesc.Texture2D.MostDetailedMip = 0;
	cacheSrvDesc.Texture2D.MipLevels = cacheTexture->GetDesc().MipLevels;
	cacheSrvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

	D3D12_CPU_DESCRIPTOR_HANDLE hdrCacheTextureHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
	hdrCacheTextureHandle.ptr += (UINT)DescriptorID::cacheBuffer * mSrvDescriptorSize;
	mDevice_v5->CreateShaderResourceView(cacheTexture, &cacheSrvDesc, hdrCacheTextureHandle);*/

	D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc = {};
	samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
	samplerHeapDesc.NumDescriptors = 1;
	samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

	ThrowIfFailed(mDevice_v5->CreateDescriptorHeap(&samplerHeapDesc, IID_PPV_ARGS(&mSamplerHeap)));

	D3D12_SAMPLER_DESC samplerDesc = {};
	samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	samplerDesc.MinLOD = 0;
	samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;

	mDevice_v5->CreateSampler(&samplerDesc, mSamplerHeap->GetCPUDescriptorHandleForHeapStart());
}

void DXRPathTracer::onSizeChanged(UINT width, UINT height)
{
	mTracerOutW = width;
	mTracerOutH = height;

	mCamera.setLens(1.f / 9.f * XM_PI, float(mTracerOutW) / mTracerOutH, 1.0f, 1000.0f);

	if (mTracerOutBuffer != nullptr)
		mTracerOutBuffer.Reset();

	UINT64 bufferSize = _bpp(mTracerOutFormat) * mTracerOutW * mTracerOutH;
	mTracerOutBuffer = createCommittedBuffer(bufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	{
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavDesc.Format = mTracerOutFormat;
		uavDesc.Buffer.NumElements = mTracerOutW * mTracerOutH;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
	uavDescriptorHandle.ptr += ((UINT)DescriptorID::outUAV) * mSrvDescriptorSize;
	mDevice_v5->CreateUnorderedAccessView(mTracerOutBuffer.Get(), nullptr, &uavDesc, uavDescriptorHandle);
}

ComPtr<ID3D12RootSignature> DXRPathTracer::buildRootSignatures(const D3D12_ROOT_SIGNATURE_DESC& desc)
{
	ComPtr<ID3DBlob> pSigBlob;
	ComPtr<ID3DBlob> pErrorBlob;
	HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &pSigBlob, &pErrorBlob);

	if (pErrorBlob)
		throw Error((char*)pErrorBlob->GetBufferPointer());

	ComPtr<ID3D12RootSignature> pRootSig;
	ThrowIfFailed(mDevice_v5->CreateRootSignature(0, pSigBlob->GetBufferPointer(), pSigBlob->GetBufferSize(), IID_PPV_ARGS(&pRootSig)));

	return pRootSig;
}

void DXRPathTracer::declareRootSignatures()
{
	//Global
	vector<D3D12_DESCRIPTOR_RANGE> globalRange;
	globalRange.resize(4);

	globalRange[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	globalRange[0].NumDescriptors = 1;
	globalRange[0].BaseShaderRegister = 0;
	globalRange[0].RegisterSpace = 0;
	globalRange[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

	globalRange[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	globalRange[1].NumDescriptors = 4;
	globalRange[1].BaseShaderRegister = 0;
	globalRange[1].RegisterSpace = 0;
	globalRange[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

	globalRange[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	globalRange[2].NumDescriptors = 2;
	globalRange[2].BaseShaderRegister = 4;
	globalRange[2].RegisterSpace = 0;
	globalRange[2].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

	globalRange[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
	globalRange[3].NumDescriptors = 1;
	globalRange[3].BaseShaderRegister = 0;
	globalRange[3].RegisterSpace = 0;
	globalRange[3].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

	vector<D3D12_ROOT_PARAMETER> globalRootParams;
	globalRootParams.resize(RootParamID::numParams);

	globalRootParams[RootParamID::tableForOutBuffer].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	globalRootParams[RootParamID::tableForOutBuffer].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	globalRootParams[RootParamID::tableForOutBuffer].DescriptorTable.NumDescriptorRanges = 1;
	globalRootParams[RootParamID::tableForOutBuffer].DescriptorTable.pDescriptorRanges = &globalRange[0];

	globalRootParams[RootParamID::pointerForAccelerationStructure].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
	globalRootParams[RootParamID::pointerForAccelerationStructure].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	globalRootParams[RootParamID::pointerForAccelerationStructure].Descriptor.ShaderRegister = 0;
	globalRootParams[RootParamID::pointerForAccelerationStructure].Descriptor.RegisterSpace = 100;

	globalRootParams[RootParamID::tableForGeometryInputs].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	globalRootParams[RootParamID::tableForGeometryInputs].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	globalRootParams[RootParamID::tableForGeometryInputs].DescriptorTable.NumDescriptorRanges = 1;
	globalRootParams[RootParamID::tableForGeometryInputs].DescriptorTable.pDescriptorRanges = &globalRange[1];

	globalRootParams[RootParamID::pointerForGlobalConstants].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
	globalRootParams[RootParamID::pointerForGlobalConstants].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	globalRootParams[RootParamID::pointerForGlobalConstants].Descriptor.ShaderRegister = 0;
	globalRootParams[RootParamID::pointerForGlobalConstants].Descriptor.RegisterSpace = 0;

	globalRootParams[RootParamID::tableForHDRMap].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	globalRootParams[RootParamID::tableForHDRMap].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	globalRootParams[RootParamID::tableForHDRMap].DescriptorTable.NumDescriptorRanges = 1;
	globalRootParams[RootParamID::tableForHDRMap].DescriptorTable.pDescriptorRanges = &globalRange[2];

	globalRootParams[RootParamID::tableForSampler].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	globalRootParams[RootParamID::tableForSampler].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	globalRootParams[RootParamID::tableForSampler].DescriptorTable.NumDescriptorRanges = 1;
	globalRootParams[RootParamID::tableForSampler].DescriptorTable.pDescriptorRanges = &globalRange[3];

	D3D12_ROOT_SIGNATURE_DESC globalRootSigDesc = {};
	globalRootSigDesc.NumParameters = RootParamID::numParams;
	globalRootSigDesc.pParameters = globalRootParams.data();
	globalRootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
	mGlobalRS = buildRootSignatures(globalRootSigDesc);

	//Local
	vector<D3D12_ROOT_PARAMETER> localRootParams;
	localRootParams.resize(HitGroupParamID::numParams);
	localRootParams[HitGroupParamID::constantForObject].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	localRootParams[HitGroupParamID::constantForObject].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	localRootParams[HitGroupParamID::constantForObject].Constants.Num32BitValues = (sizeof(ObjectConstants) + 3) / 4;
	localRootParams[HitGroupParamID::constantForObject].Constants.ShaderRegister = 1;
	localRootParams[HitGroupParamID::constantForObject].Constants.RegisterSpace = 0;

	D3D12_ROOT_SIGNATURE_DESC localRootSigDesc = {};
	localRootSigDesc.NumParameters = HitGroupParamID::numParams;
	localRootSigDesc.pParameters = localRootParams.data();
	localRootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
	mHitGroupRS = buildRootSignatures(localRootSigDesc);
}

void DXRPathTracer::buildRaytracingPipeline()
{
	vector<D3D12_STATE_SUBOBJECT> subObjects;
	subObjects.resize(7);
	UINT index = 0;

	//Global Root Signature
	D3D12_STATE_SUBOBJECT subObjGlobalRS = {};

	D3D12_GLOBAL_ROOT_SIGNATURE grsDesc;
	grsDesc.pGlobalRootSignature = mGlobalRS.Get();
	subObjGlobalRS.pDesc = (void*)&grsDesc;
	subObjGlobalRS.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;

	subObjects[index++] = subObjGlobalRS;

	//Raytracing Pipeline Config
	D3D12_STATE_SUBOBJECT subObjPipelineCfg = {};

	D3D12_RAYTRACING_PIPELINE_CONFIG pipelineCfg;
	pipelineCfg.MaxTraceRecursionDepth = D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH;
	subObjPipelineCfg.pDesc = &pipelineCfg;
	subObjPipelineCfg.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;

	subObjects[index++] = subObjPipelineCfg;

	//DXIL Library
	D3D12_STATE_SUBOBJECT subObjDXILLib = {};

	D3D12_EXPORT_DESC rayGenExportDesc[4] = {};
	rayGenExportDesc[0].Name = cRayGenShaderName;
	rayGenExportDesc[0].Flags = D3D12_EXPORT_FLAG_NONE;

	rayGenExportDesc[1].Name = cClosestHitshaderName;
	rayGenExportDesc[1].Flags = D3D12_EXPORT_FLAG_NONE;

	rayGenExportDesc[2].Name = cMissShaderName;
	rayGenExportDesc[2].Flags = D3D12_EXPORT_FLAG_NONE;

	rayGenExportDesc[3].Name = cMissShadowShaderName;
	rayGenExportDesc[3].Flags = D3D12_EXPORT_FLAG_NONE;

	mDxrLib.load(L"DXRShader.cso");
	D3D12_DXIL_LIBRARY_DESC dxilLibDesc = {};
	dxilLibDesc.DXILLibrary = mDxrLib.getCode();
	dxilLibDesc.NumExports = _countof(rayGenExportDesc);
	dxilLibDesc.pExports = rayGenExportDesc;
	subObjDXILLib.pDesc = (void*)&dxilLibDesc;
	subObjDXILLib.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;

	subObjects[index++] = subObjDXILLib;

	//HitGroup Shader Table Group
	D3D12_STATE_SUBOBJECT subObjHitGroup = {};

	D3D12_HIT_GROUP_DESC hitGroupDesc;
	hitGroupDesc.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
	hitGroupDesc.ClosestHitShaderImport = cClosestHitshaderName;
	hitGroupDesc.AnyHitShaderImport = nullptr;
	hitGroupDesc.HitGroupExport = cHitGroupName;
	hitGroupDesc.IntersectionShaderImport = nullptr;
	subObjHitGroup.pDesc = (void*)&hitGroupDesc;
	subObjHitGroup.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;

	subObjects[index++] = subObjHitGroup;

	//Raytracing Shader Config
	D3D12_STATE_SUBOBJECT subObjShaderCfg = {};

	D3D12_RAYTRACING_SHADER_CONFIG shaderCfg = {};
	shaderCfg.MaxPayloadSizeInBytes = 80;
	shaderCfg.MaxAttributeSizeInBytes = 8;
	subObjShaderCfg.pDesc = (void*)&shaderCfg;
	subObjShaderCfg.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;

	subObjects[index++] = subObjShaderCfg;

	//Local Root Signature
	D3D12_STATE_SUBOBJECT subObjLocalRS = {};

	D3D12_LOCAL_ROOT_SIGNATURE lrsDesc;
	lrsDesc.pLocalRootSignature = mHitGroupRS.Get();
	subObjLocalRS.pDesc = (void*)&lrsDesc;
	subObjLocalRS.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;

	UINT localObjIdx = index;
	subObjects[index++] = subObjLocalRS;

	//Association
	D3D12_STATE_SUBOBJECT subObjAssoc = {};

	vector<LPCWSTR> exportName;
	exportName.resize(3);
	exportName[0] = cHitGroupName;
	exportName[1] = cMissShaderName;
	exportName[2] = cMissShadowShaderName;

	D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION obj2ExportsAssoc = {};
	obj2ExportsAssoc.pSubobjectToAssociate = &subObjects[localObjIdx];
	obj2ExportsAssoc.NumExports = (UINT)exportName.size();
	obj2ExportsAssoc.pExports = exportName.data();

	subObjAssoc.pDesc = &obj2ExportsAssoc;
	subObjAssoc.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;

	subObjects[index++] = subObjAssoc;

	assert(index == subObjects.size());

	//Create Raytracing PSO
	D3D12_STATE_OBJECT_DESC raytracingPSODesc = {};
	raytracingPSODesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
	raytracingPSODesc.NumSubobjects = (UINT)subObjects.size();
	raytracingPSODesc.pSubobjects = subObjects.data();

	ThrowIfFailed(mDevice_v5->CreateStateObject(&raytracingPSODesc, IID_PPV_ARGS(&mRTPipeline)));
}

void DXRPathTracer::initializeApplication()
{
	mCamera.setLens(1.f / 9.f * XM_PI, float(mTracerOutW) / mTracerOutH, 1.0f, 1000.0f);
	
	mGlobalConstantsBuffer = createCommittedBuffer(sizeof(GlobalConstants));

	mMaxBufferSize = _bpp(mTracerOutFormat) * 1920 * 1080;
	mMaxBufferSize = _align(mMaxBufferSize, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
	mReadBackBuffer = createCommittedBuffer(mMaxBufferSize, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
}

void DXRPathTracer::update()
{
	mCamera.update();

	if (mCamera.notifyChanged())
	{
		mGlobalConstants.cameraPos = mCamera.getPosition3f();
		mGlobalConstants.accumulatedFrame = 0;
		mGlobalConstants.hdrResolution = mHdrResolution;

		XMMATRIX view = mCamera.getView();
		XMMATRIX proj = mCamera.getProj();

		XMMATRIX viewProj = XMMatrixMultiply(view, proj);
		XMMATRIX invViewProj = XMMatrixInverse(&XMMatrixDeterminant(viewProj), viewProj);
		XMStoreFloat4x4(&mGlobalConstants.invViewProj, XMMatrixTranspose(invViewProj));
	}
	else
		mGlobalConstants.accumulatedFrame++;

	UINT8* pGlobalConstants;
	ThrowIfFailed(mGlobalConstantsBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pGlobalConstants)));
	memcpy(pGlobalConstants, &mGlobalConstants, sizeof(GlobalConstants));
}

TracedResult DXRPathTracer::shootRays()
{
	D3D12_DISPATCH_RAYS_DESC desc = {};
	//rayGen
	desc.RayGenerationShaderRecord.StartAddress = mRayGenShaderTable->GetGPUVirtualAddress();
	desc.RayGenerationShaderRecord.SizeInBytes = mRayGenShaderTable->GetDesc().Width;

	//miss
	desc.MissShaderTable.StartAddress = mMissShaderTable->GetGPUVirtualAddress();
	desc.MissShaderTable.StrideInBytes = _align(D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
	desc.MissShaderTable.SizeInBytes = mMissShaderTable->GetDesc().Width;

	//hit
	desc.HitGroupTable.StartAddress = mHitGroupShaderTable->GetGPUVirtualAddress();
	desc.HitGroupTable.StrideInBytes = _align(D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES + (UINT)sizeof(ObjectConstants), D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
	desc.HitGroupTable.SizeInBytes = mHitGroupShaderTable->GetDesc().Width;

	desc.Width = mTracerOutW;
	desc.Height = mTracerOutH;
	desc.Depth = 1;

	mCmdList_v4->SetPipelineState1(mRTPipeline.Get());
	ID3D12DescriptorHeap* ppDescriptorHeaps[] = { mSrvUavHeap.Get(), mSamplerHeap.Get() };
	mCmdList_v4->SetDescriptorHeaps(2, ppDescriptorHeaps);
	mCmdList_v4->SetComputeRootSignature(mGlobalRS.Get());

	//tableForOutBuffer
	D3D12_GPU_DESCRIPTOR_HANDLE objUAVHandle = mSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
	objUAVHandle.ptr += (UINT)DescriptorID::outUAV * mSrvDescriptorSize;
	mCmdList_v4->SetComputeRootDescriptorTable((UINT)RootParamID::tableForOutBuffer, objUAVHandle);

	//pointerForAccelerationStructure
	mCmdList_v4->SetComputeRootShaderResourceView((UINT)RootParamID::pointerForAccelerationStructure, mTopLevelAccelerationStructure->GetGPUVirtualAddress());

	//tableForGeometryInputs
	D3D12_GPU_DESCRIPTOR_HANDLE objGeometryInputsHandle = mSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
	objGeometryInputsHandle.ptr += (UINT)DescriptorID::sceneObjectBuff * mSrvDescriptorSize;
	mCmdList_v4->SetComputeRootDescriptorTable((UINT)RootParamID::tableForGeometryInputs, objGeometryInputsHandle);

	//pointerForGlobalConstants
	mCmdList_v4->SetComputeRootConstantBufferView((UINT)RootParamID::pointerForGlobalConstants, mGlobalConstantsBuffer->GetGPUVirtualAddress());

	//tableForHDRMap
	D3D12_GPU_DESCRIPTOR_HANDLE hdrMapHandle = mSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
	hdrMapHandle.ptr += ((UINT)DescriptorID::hdrBuffer * mSrvDescriptorSize);
	mCmdList_v4->SetComputeRootDescriptorTable((UINT)RootParamID::tableForHDRMap, hdrMapHandle);

	//tableForSampler
	D3D12_GPU_DESCRIPTOR_HANDLE samplerHandle = mSamplerHeap->GetGPUDescriptorHandleForHeapStart();
	mCmdList_v4->SetComputeRootDescriptorTable((UINT)RootParamID::tableForSampler, samplerHandle);

	mCmdList_v4->DispatchRays(&desc);

	D3D12_RESOURCE_DESC tracerBufferDesc = mTracerOutBuffer->GetDesc();
	if (tracerBufferDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
	{
		if (tracerBufferDesc.Width > mMaxBufferSize)
		{
			mMaxBufferSize = _align(tracerBufferDesc.Width * 2, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
			mReadBackBuffer = createCommittedBuffer(mMaxBufferSize, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
		}

		mCmdList_v4->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mTracerOutBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
		mCmdList_v4->CopyBufferRegion(mReadBackBuffer.Get(), 0, mTracerOutBuffer.Get(), 0, tracerBufferDesc.Width);
		mCmdList_v4->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mTracerOutBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
	}

	ThrowIfFailed(mCmdList_v4->Close());
	ID3D12CommandList* cmdLists[] = { mCmdList_v4.Get() };
	mCmdQueue_v0->ExecuteCommandLists(1, cmdLists);
	mFence_v0.waitCommandQueue(mCmdQueue_v0.Get());
	ThrowIfFailed(mCmdAllocator_v0->Reset());
	ThrowIfFailed(mCmdList_v4->Reset(mCmdAllocator_v0.Get(), nullptr));

	UINT8* tracedResultData;
	ThrowIfFailed(mReadBackBuffer->Map(0, nullptr, reinterpret_cast<void**>(&tracedResultData)));
	TracedResult result;
	result.data = tracedResultData;
	result.width = mTracerOutW;
	result.height = mTracerOutH;
	result.pixelSize = _bpp(mTracerOutFormat);

	mReadBackBuffer->Unmap(0, nullptr);

	return result;
}

void DXRPathTracer::setupShaderTable()
{
	void* pRaygenShaderIdentifier;
	void* pMissShaderIdentifier;
	void* pMissShadowShaderIdentifier;
	void* pHitGroupShaderIdentifier;

	UINT numObjs = mScene->numObjects();

	ComPtr<ID3D12StateObjectProperties> pStateObjectProperties;
	ThrowIfFailed(mRTPipeline->QueryInterface(IID_PPV_ARGS(&pStateObjectProperties)));

	pRaygenShaderIdentifier = pStateObjectProperties->GetShaderIdentifier(cRayGenShaderName);
	pMissShaderIdentifier = pStateObjectProperties->GetShaderIdentifier(cMissShaderName);
	pMissShadowShaderIdentifier = pStateObjectProperties->GetShaderIdentifier(cMissShadowShaderName);
	pHitGroupShaderIdentifier = pStateObjectProperties->GetShaderIdentifier(cHitGroupName);

	D3D12_HEAP_DESC uploadHeapDesc = {};
	UINT64 n64HeapSize = 1024 * 1024;
	UINT64 n64HeapOffset = 0;
	UINT64 n64AllocSize = 0;
	UINT8* pBufs = nullptr;

	uploadHeapDesc.SizeInBytes = _align(n64HeapSize, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
	uploadHeapDesc.Alignment = 0;
	uploadHeapDesc.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
	uploadHeapDesc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
	uploadHeapDesc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
	uploadHeapDesc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

	ThrowIfFailed(mDevice_v5->CreateHeap(&uploadHeapDesc, IID_PPV_ARGS(&mShaderTableHeap_v1)));

	//rayGen shader table
	{
		UINT nNumShaderRecords = 1;
		UINT nShaderRecordSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

		n64AllocSize = nNumShaderRecords * nShaderRecordSize;
		n64AllocSize = _align(n64AllocSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

		mRayGenShaderTable = createPlacedBuffer(mShaderTableHeap_v1.Get(), n64HeapOffset, n64AllocSize, D3D12_RESOURCE_STATE_GENERIC_READ);
		ThrowIfFailed(mRayGenShaderTable->Map(0, &CD3DX12_RANGE(0, 0), reinterpret_cast<void**>(&pBufs)));

		memcpy(pBufs, pRaygenShaderIdentifier, nShaderRecordSize);

		mRayGenShaderTable->Unmap(0, nullptr);
	}

	n64HeapOffset += _align(n64AllocSize, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
	ThrowIfFalse(n64HeapOffset < n64HeapSize);

	//miss shader table
	{
		UINT nNumShaderRecords = 2;
		UINT nShaderRecordSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
		n64AllocSize = nNumShaderRecords * nShaderRecordSize;
		n64AllocSize = _align(n64AllocSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

		mMissShaderTable = createPlacedBuffer(mShaderTableHeap_v1.Get(), n64HeapOffset, n64AllocSize, D3D12_RESOURCE_STATE_GENERIC_READ);
		pBufs = nullptr;
		ThrowIfFailed(mMissShaderTable->Map(0, &CD3DX12_RANGE(0, 0), reinterpret_cast<void**>(&pBufs)));

		memcpy(pBufs, pMissShaderIdentifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		memcpy(pBufs + nShaderRecordSize, pMissShadowShaderIdentifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);

		mMissShaderTable->Unmap(0, nullptr);
	}

	n64HeapOffset += _align(n64AllocSize, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
	ThrowIfFailed(n64HeapOffset < n64HeapSize);

	//hit group shader table
	{
		UINT nNumShaderRecords = numObjs;
		UINT nShaderRecordSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES + sizeof(ObjectConstants);
		nShaderRecordSize = _align(nShaderRecordSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
		n64AllocSize = nNumShaderRecords * nShaderRecordSize;

		mHitGroupShaderTable = createPlacedBuffer(mShaderTableHeap_v1.Get(), n64HeapOffset, n64AllocSize, D3D12_RESOURCE_STATE_GENERIC_READ);
		pBufs = nullptr;
		ThrowIfFailed(mHitGroupShaderTable->Map(0, &CD3DX12_RANGE(0, 0), reinterpret_cast<void**>(&pBufs)));

		objConsts.resize(numObjs);
		for (UINT i = 0; i < numObjs; ++i)
		{
			objConsts[i].objectIdx = i;

			memcpy(pBufs + nShaderRecordSize * i, pHitGroupShaderIdentifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
			memcpy(pBufs + nShaderRecordSize * i + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, &objConsts[i], sizeof(ObjectConstants));
		}

		mHitGroupShaderTable->Unmap(0, nullptr);
	}
}

ComPtr<ID3D12Resource> DXRPathTracer::createAS(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& buildInput, ComPtr<ID3D12Resource>* scrach)
{
	ComPtr<ID3D12Resource> AS;

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
	mDevice_v5->GetRaytracingAccelerationStructurePrebuildInfo(&buildInput, &info);

	*scrach = createCommittedBuffer(info.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	AS = createCommittedBuffer(info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
	asDesc.Inputs = buildInput;
	asDesc.DestAccelerationStructureData = AS->GetGPUVirtualAddress();
	asDesc.ScratchAccelerationStructureData = (*scrach)->GetGPUVirtualAddress();
	mCmdList_v4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);

	D3D12_RESOURCE_BARRIER uavBarrier = {};
	uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	uavBarrier.UAV.pResource = AS.Get();
	mCmdList_v4->ResourceBarrier(1, &uavBarrier);

	return AS;
}

void DXRPathTracer::buildBLAS(
	ComPtr<ID3D12Resource>* blas,
	ComPtr<ID3D12Resource>* scrach,
	const GPUMesh gpuMeshArr[],
	UINT numMeshes,
	UINT vertexStride,
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags)
{
	D3D12_RAYTRACING_GEOMETRY_DESC meshDesc = {};
	meshDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
	meshDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
	meshDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
	meshDesc.Triangles.VertexBuffer.StrideInBytes = vertexStride;
	meshDesc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;

	vector<D3D12_RAYTRACING_GEOMETRY_DESC> geoDesc(numMeshes);
	for (UINT i = 0; i < numMeshes; ++i)
	{
		geoDesc[i] = meshDesc;
		geoDesc[i].Triangles.VertexCount = gpuMeshArr[i].numVertices;
		geoDesc[i].Triangles.VertexBuffer.StartAddress = gpuMeshArr[i].vertexBufferVA;
		geoDesc[i].Triangles.IndexCount = gpuMeshArr[i].numTridices * 3;
		geoDesc[i].Triangles.IndexBuffer = gpuMeshArr[i].tridexBufferVA;
	}

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS buildInput = {};
	buildInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
	buildInput.NumDescs = numMeshes;
	buildInput.Flags = buildFlags;
	buildInput.pGeometryDescs = geoDesc.data();

	*blas = createAS(buildInput, scrach);
}

void DXRPathTracer::buildTLAS(
	ComPtr<ID3D12Resource>* tlas,
	ComPtr<ID3D12Resource>* scrach,
	ComPtr<ID3D12Resource>* instanceDescArr,
	ID3D12Resource* const blasArr[],
	const dxTransform transformArr[],
	UINT numBlas,
	UINT instanceMultiplier,
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags)
{
	*instanceDescArr = createCommittedBuffer(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * numBlas);

	D3D12_RAYTRACING_INSTANCE_DESC* pInsDescArr;
	(*instanceDescArr)->Map(0, nullptr, (void**)&pInsDescArr);
	{
		memset(pInsDescArr, 0, sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * numBlas);
		for (UINT i = 0; i < numBlas; ++i)
		{
			pInsDescArr[i].InstanceMask = 0xFF;
			pInsDescArr[i].InstanceContributionToHitGroupIndex = i * instanceMultiplier;
			*(dxTransform*)(pInsDescArr[i].Transform) = transformArr[i];
			pInsDescArr[i].AccelerationStructure = const_cast<ID3D12Resource*>(blasArr[i])->GetGPUVirtualAddress();
		}
	}
	(*instanceDescArr)->Unmap(0, nullptr);

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS buildInput = {};
	buildInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
	buildInput.NumDescs = numBlas;
	buildInput.Flags = buildFlags;
	buildInput.InstanceDescs = (*instanceDescArr)->GetGPUVirtualAddress();

	*tlas = createAS(buildInput, scrach);
}

void DXRPathTracer::buildAccelerationStructure()
{
	UINT numObjs = mScene->numObjects();
	vector<GPUMesh> gpuMeshArr(numObjs);
	vector<dxTransform> transformArr(numObjs);

	D3D12_GPU_VIRTUAL_ADDRESS vtxAddr = mVertexBuffer->GetGPUVirtualAddress();
	D3D12_GPU_VIRTUAL_ADDRESS tdxAddr = mIndexBuffer->GetGPUVirtualAddress();
	for (UINT objIdx = 0; objIdx < numObjs; ++objIdx)
	{
		const SceneObject& obj = mScene->getObject(objIdx);

		gpuMeshArr[objIdx].numVertices = obj.numVertices;
		gpuMeshArr[objIdx].vertexBufferVA = vtxAddr + obj.vertexOffset * sizeof(Vertex);
		gpuMeshArr[objIdx].numTridices = obj.numTridices;
		gpuMeshArr[objIdx].tridexBufferVA = tdxAddr + obj.tridexOffset * sizeof(Tridex);

		transformArr[objIdx] = obj.modelMatrix;
	}

	assert(mTopLevelAccelerationStructure == nullptr);
	assert(gpuMeshArr.size() == transformArr.size());

	UINT numObjsPerBlas = 1;
	UINT numBottomLevels = numObjs;
	mBottomLevelAccelerationStructure.resize(numBottomLevels, nullptr);
	Scratch.resize(numBottomLevels + 1, nullptr);

	for (UINT i = 0; i < numBottomLevels; ++i)
	{
		buildBLAS(&mBottomLevelAccelerationStructure[i], &Scratch[i], &gpuMeshArr[i], numObjsPerBlas, sizeof(Vertex), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE);
	}

	const vector<dxTransform>* topLevelTransform;
	vector<dxTransform> identityArr;
	topLevelTransform = &transformArr;

	buildTLAS(&mTopLevelAccelerationStructure, &Scratch[numBottomLevels], &InstanceDesc, mBottomLevelAccelerationStructure[0].GetAddressOf(), &(*topLevelTransform)[0], numBottomLevels, 1, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE);

	ThrowIfFailed(mCmdList_v4->Close());
	ID3D12CommandList* cmdLists[] = { mCmdList_v4.Get() };
	mCmdQueue_v0->ExecuteCommandLists(1, cmdLists);
	mFence_v0.waitCommandQueue(mCmdQueue_v0.Get());
	ThrowIfFailed(mCmdAllocator_v0->Reset());
	ThrowIfFailed(mCmdList_v4->Reset(mCmdAllocator_v0.Get(), nullptr));
}

void DXRPathTracer::setupScene(const Scene* scene)
{
	UINT numObjs = scene->numObjects();

	const vector<Vertex> vtxArr = scene->getVertexArray();
	const vector<Tridex> tdxArr = scene->getTridexArray();
	const vector<Material> mtlArr = scene->getMaterialArray();

	UINT64 vtxBuffSize = vtxArr.size() * sizeof(Vertex);
	UINT64 tdxBuffSize = tdxArr.size() * sizeof(Tridex);
	UINT64 mtlBuffSize = mtlArr.size() * sizeof(Material);
	UINT64 objBuffSize = numObjs * sizeof(GPUSceneObject);

	ComPtr<ID3D12Resource> uploader = createCommittedBuffer(
		vtxBuffSize + tdxBuffSize + mtlBuffSize + objBuffSize);
	UINT64 uploaderOffset = 0;

	auto initBuffer = [&](ComPtr<ID3D12Resource>& buff, UINT64 buffSize, void* srcData)
	{
		if (buffSize == 0)
			return;
		buff = createCommittedBuffer(buffSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);

		UINT8* pBufs = nullptr;
		uploader->Map(0, &CD3DX12_RANGE(0, 0), reinterpret_cast<void**>(&pBufs));
		memcpy(pBufs + uploaderOffset, srcData, buffSize);
		mCmdList_v4->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(buff.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
		mCmdList_v4->CopyBufferRegion(buff.Get(), 0, uploader.Get(), uploaderOffset, buffSize);
		mCmdList_v4->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(buff.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));
		uploaderOffset += buffSize;
	};

	initBuffer(mVertexBuffer, vtxBuffSize, (void*)vtxArr.data());
	initBuffer(mIndexBuffer, tdxBuffSize, (void*)tdxArr.data());
	initBuffer(mMaterialBuffer, mtlBuffSize, (void*)mtlArr.data());

	mSceneObjectBuffer = createCommittedBuffer(objBuffSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);

	void* cpuAddress;
	uploader->Map(0, &CD3DX12_RANGE(0, 0), &cpuAddress);
	GPUSceneObject* copyDst = (GPUSceneObject*)((UINT8*)cpuAddress + uploaderOffset);
	for (UINT objIdx = 0; objIdx < numObjs; ++objIdx)
	{
		const SceneObject& obj = scene->getObject(objIdx);

		GPUSceneObject gpuObj = {};
		gpuObj.vertexOffset = obj.vertexOffset;
		gpuObj.tridexOffset = obj.tridexOffset;
		gpuObj.materialIdx = obj.materialIdx;
		gpuObj.modelMatrix = obj.modelMatrix;

		copyDst[objIdx] = gpuObj;
	}

	mCmdList_v4->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mSceneObjectBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
	mCmdList_v4->CopyBufferRegion(mSceneObjectBuffer.Get(), 0, uploader.Get(), uploaderOffset, objBuffSize);
	mCmdList_v4->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mSceneObjectBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));

	ThrowIfFailed(mCmdList_v4->Close());
	ID3D12CommandList* cmdLists[] = { mCmdList_v4.Get() };
	mCmdQueue_v0->ExecuteCommandLists(1, cmdLists);
	mFence_v0.waitCommandQueue(mCmdQueue_v0.Get());
	ThrowIfFailed(mCmdAllocator_v0->Reset());
	ThrowIfFailed(mCmdList_v4->Reset(mCmdAllocator_v0.Get(), nullptr));

	this->mScene = const_cast<Scene*>(scene);

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	//SceneObjectBuffer
	{
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Buffer.StructureByteStride = sizeof(GPUSceneObject);
		srvDesc.Buffer.NumElements = numObjs;
	}
	D3D12_CPU_DESCRIPTOR_HANDLE sceneObjectHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
	sceneObjectHandle.ptr += (UINT)DescriptorID::sceneObjectBuff * mSrvDescriptorSize;
	mDevice_v5->CreateShaderResourceView(mSceneObjectBuffer.Get(), &srvDesc, sceneObjectHandle);

	//Vertex srv
	{
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.Buffer.NumElements = (UINT)vtxArr.size();
		srvDesc.Buffer.StructureByteStride = sizeof(Vertex);
	}
	D3D12_CPU_DESCRIPTOR_HANDLE vertexSrvHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
	vertexSrvHandle.ptr += UINT(DescriptorID::vertexBuff) * mSrvDescriptorSize;
	mDevice_v5->CreateShaderResourceView(mVertexBuffer.Get(), &srvDesc, vertexSrvHandle);

	//Index srv
	{
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = DXGI_FORMAT_R32G32B32_UINT;
		srvDesc.Buffer.NumElements = (UINT)tdxArr.size();
		srvDesc.Buffer.StructureByteStride = 0;
	}
	D3D12_CPU_DESCRIPTOR_HANDLE indexSrvHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
	indexSrvHandle.ptr += (UINT)DescriptorID::tridexBuff * mSrvDescriptorSize;
	mDevice_v5->CreateShaderResourceView(mIndexBuffer.Get(), &srvDesc, indexSrvHandle);

	//MaterialBuffer
	{
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Buffer.StructureByteStride = sizeof(Material);
		srvDesc.Buffer.NumElements = (UINT)mtlArr.size();
	}
	D3D12_CPU_DESCRIPTOR_HANDLE cpuMaterialBuffHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
	cpuMaterialBuffHandle.ptr += (UINT)DescriptorID::materialBuff * mSrvDescriptorSize;
	mDevice_v5->CreateShaderResourceView(mMaterialBuffer.Get(), &srvDesc, cpuMaterialBuffHandle);

	setupShaderTable();

	buildAccelerationStructure();
}