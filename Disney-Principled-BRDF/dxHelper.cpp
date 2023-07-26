#include "dxHelper.h"

static ComPtr<IDXGIFactory2> gFactory_v2 = nullptr;
static ComPtr<ID3D12Device> gDevice_v0 = nullptr;
static ComPtr<IDXGIAdapter> gAdapter_v0 = nullptr;

ComPtr<IDXGIFactory2> getFactory()
{
	if (!gFactory_v2)
	{
#ifdef _DEBUG
		ComPtr<ID3D12Debug> debuger;
		if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debuger))))
			debuger->EnableDebugLayer();
#endif
		ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&gFactory_v2)));
	}

	return gFactory_v2;
}

ComPtr<IDXGIAdapter> getRTXAdapter()
{
	if (!gAdapter_v0)
	{
		for (UINT i = 0; DXGI_ERROR_NOT_FOUND != getFactory()->EnumAdapters(i, &gAdapter_v0); i++)
		{
			ComPtr<ID3D12Device> tempDevice_v0;
			ThrowIfFailed(D3D12CreateDevice(gAdapter_v0.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&tempDevice_v0)));

			D3D12_FEATURE_DATA_D3D12_OPTIONS5 features5;
			HRESULT hr = tempDevice_v0->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &features5, sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS5));

			if (FAILED(hr))
			{
				throw Error("Your Window 10 version must be at least 1809 to support D3D12_FEATURE_D3D12_OPTIONS5.");
			}

			if (features5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
			{
				return gAdapter_v0;
			}
		}

		throw Error("There are not DXR supported adapters(video cards such as Nvidia's Volta or Turing RTX).Also, The DXR fallback layer is not supported in this application.");
	}

	return gAdapter_v0;
}

ComPtr<ID3D12Device> createDX12Device(IDXGIAdapter* adapter)
{
	ThrowIfFailed(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&gDevice_v0)));

	return gDevice_v0;
}

ComPtr<ID3D12Resource> createCommittedBuffer(UINT64 bufferSize, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_FLAGS resourceFlags, D3D12_RESOURCE_STATES resourceStates)
{
	ComPtr<ID3D12Resource> resource;
	ThrowIfFailed(gDevice_v0->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(heapType),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(bufferSize, resourceFlags),
		resourceStates,
		nullptr,
		IID_PPV_ARGS(&resource)));

	return resource;
}

ComPtr<ID3D12Resource> createPlacedBuffer(ID3D12Heap1* heap, UINT64 heapOffset, UINT64 bufferSize, D3D12_RESOURCE_STATES state)
{
	ComPtr<ID3D12Resource> resource;
	ThrowIfFailed(gDevice_v0->CreatePlacedResource(
		heap,
		heapOffset,
		&CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
		state,
		nullptr,
		IID_PPV_ARGS(&resource)));

	return resource;
}

ComPtr<ID3D12Resource> createDefaultTexture(DXGI_FORMAT format, UINT width, UINT height, D3D12_RESOURCE_STATES state)
{
	ComPtr<ID3D12Resource> resource;
	ThrowIfFailed(gDevice_v0->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Tex2D(format, width, height),
		state,
		nullptr,
		IID_PPV_ARGS(&resource)));

	return resource;
}

static inline UINT length(const WCHAR* wstr)
{
	UINT idx = 0;
	for(const WCHAR* ch = wstr; *ch != L'\0'; ++ch)
		idx++;

	return idx;
}

static inline void copy(WCHAR* dst, const WCHAR* src, UINT num)
{
	for (UINT i = 0; i < num; ++i)
	{
		dst[i] = src[i];
	}
}

static inline UINT addStr(WCHAR* dst, const WCHAR* src1, const WCHAR* src2)
{
	UINT l1 = length(src1);
	UINT l2 = length(src2);

	copy(dst, src1, l1);
	copy(dst + l1, src2, l2);
	dst[l1 + l2] = L'\0';

	return l1 + l2;
}

const LPCWSTR dxShader::csoFolder = L"bin/cso/";
const LPCWSTR dxShader::hlslFolder = L"";

void dxShader::flush()
{
}

void dxShader::load(LPCWSTR csoFile)
{
	WCHAR filePath[256];
	addStr(filePath, csoFolder, csoFile);
	ThrowIfFailed(D3DReadFileToBlob(filePath, &mCode));
}

void dxShader::load(LPCWSTR hlslFile, const char* entryFtn, const char* target)
{
	WCHAR filePath[256];
	addStr(filePath, hlslFolder, hlslFile);

	ComPtr<ID3DBlob> error = nullptr;
	UINT compileFlags = 0;
#if defined(_DEBUG)
	compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
	ThrowIfFailed(D3DCompileFromFile(filePath, nullptr, nullptr, entryFtn, target, compileFlags, 0, &mCode, &error));

	if (error)
	{
		throw Error((char*)error->GetBufferSize());
	}
}