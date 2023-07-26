#pragma once
#include "pch.h"
#include "error.h"

#define SAFE_DELETE(x) if(x) delete x; x=nullptr
#define SAFE_DELETE_ARR(x) if(x) delete[] x; x=nullptr
#define SAFE_RELEASE(x) if(x) (x)->Release(); x=nullptr

struct TracedResult
{
	void* data;
	UINT width;
	UINT height;
	UINT pixelSize;
};

struct Texture
{
	std::string name;
	char* fileName;

	FLOAT* data = nullptr;
	UINT width = 0;
	UINT height = 0;
	UINT components = 0;
	DXGI_FORMAT format;

	Microsoft::WRL::ComPtr<ID3D12Resource> resource = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> uploadResource = nullptr;
};

inline constexpr UINT _bpp(DXGI_FORMAT format)
{
	switch (format)
	{
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
		return 4;

	case DXGI_FORMAT_R32G32B32A32_FLOAT:
		return 16;

	default:
		assert(true);
		return 0;
	}
}

template<typename UnsignedType, typename PowerOfTwo>
inline constexpr UnsignedType _align(UnsignedType val, PowerOfTwo base)
{
	return (val + (UnsignedType)base - 1) & ~((UnsignedType)base - 1);
}

inline constexpr UINT64 _textureDataSize(DXGI_FORMAT format, UINT width, UINT height)
{
	const UINT64 rowSize = _bpp(format) * width;
	const UINT64 rowPitch = _align(rowSize, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
	const UINT64 dataSize = (height - 1) * rowPitch + rowSize;
	return dataSize;
}

inline void memcpyPitch(void* dstData, UINT64 dstPitch, void* srcData, UINT64 srcPitch, UINT64 rowSize, UINT numRows)
{
	assert(dstPitch >= rowSize && srcPitch >= rowSize);

	if (dstPitch == srcPitch)
	{
		memcpy(dstData, srcData, srcPitch * numRows);
	}
	else
	{
		for (UINT j = 0; j < numRows; ++j)
		{
			memcpy((UINT8*)dstData + j * dstPitch, (UINT8*)srcData + j * srcPitch, rowSize);
		}
	}
}

inline void memcpyPitch(void* dstData, UINT64 dstPitch, void* srcData, UINT64 srcPitch, UINT numRows)
{
	memcpyPitch(dstData, dstPitch, srcData, srcPitch, srcPitch, numRows);
}

ComPtr<IDXGIFactory2> getFactory();
ComPtr<IDXGIAdapter> getRTXAdapter();
ComPtr<ID3D12Device> createDX12Device(IDXGIAdapter* adapter);

ComPtr<ID3D12Resource> createCommittedBuffer(
	UINT64 bufferSize,
	D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_UPLOAD,
	D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_NONE,
	D3D12_RESOURCE_STATES resourceStates = D3D12_RESOURCE_STATE_GENERIC_READ);

ComPtr<ID3D12Resource> createPlacedBuffer(
	ID3D12Heap1* heap,
	UINT64 heapOffset,
	UINT64 bufferSize,
	D3D12_RESOURCE_STATES state);

ComPtr<ID3D12Resource> createDefaultTexture(
	DXGI_FORMAT format,
	UINT width,
	UINT height,
	D3D12_RESOURCE_STATES state);

class dxShader
{
	ComPtr<ID3DBlob> mCode;

	static const LPCWSTR csoFolder;
	static const LPCWSTR hlslFolder;

public:
	~dxShader() { flush(); }
	dxShader() {}
	dxShader(LPCWSTR csoFile) { load(csoFile); }
	dxShader(LPCWSTR hlslFile, const char* entryFtn, const char* target) { load(hlslFile, entryFtn, target); }

	D3D12_SHADER_BYTECODE getCode() const
	{
		D3D12_SHADER_BYTECODE shadercode;
		shadercode.pShaderBytecode = mCode->GetBufferPointer();
		shadercode.BytecodeLength = mCode->GetBufferSize();
		return shadercode;
	}

	void flush();
	void load(LPCWSTR csoFile);
	void load(LPCWSTR hlslFile, const char* entryPoint, const char* target);
};

class BinaryFence
{
	ComPtr<ID3D12Fence> mFence;
	HANDLE mFenceEvent = nullptr;
	UINT mFenceValue = 0u;

public:
	~BinaryFence()
	{
		if (mFenceEvent)
			CloseHandle(mFenceEvent);
	}

	void create(ID3D12Device* device)
	{
		ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFence)));
		mFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	}

	void waitCommandQueue(ID3D12CommandQueue* cmdQueue)
	{
		mFenceValue++;

		ThrowIfFailed(cmdQueue->Signal(mFence.Get(), mFenceValue));

		if (mFence->GetCompletedValue() < mFenceValue)
		{
			HANDLE eventHandle = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);

			ThrowIfFailed(mFence->SetEventOnCompletion(mFenceValue, eventHandle));

			WaitForSingleObject(eventHandle, INFINITE);
			CloseHandle(eventHandle);
		}
	}
};
