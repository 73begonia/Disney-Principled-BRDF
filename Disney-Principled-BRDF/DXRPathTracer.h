#pragma once
#include "dxHelper.h"
#include "Camera.h"
#include "Scene.h"

using pFloat4 = float(*)[4];
struct dxTransform
{
	float mat[3][4];
	dxTransform() {}
	dxTransform(float scale) : mat{}
	{
		mat[0][0] = mat[1][1] = mat[2][2] = scale;
	}
	void operator=(const Transform& tm)
	{
		*this = *(dxTransform*)&tm;
	}
	pFloat4 data() { return mat; }
};

#define NextAlignedLine __declspec(align(16))

struct GlobalConstants
{
	NextAlignedLine
	XMFLOAT3 cameraPos;
	UINT accumulatedFrame;
	NextAlignedLine
	XMFLOAT4X4 invViewProj;
	NextAlignedLine
	int hdrResolution;
};

struct ObjectConstants
{
	UINT objectIdx;
};

class DXRPathTracer
{
	UINT mTracerOutW;
	UINT mTracerOutH;
	HWND mTargetWindow;

	static const DXGI_FORMAT mTracerOutFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;

	ComPtr<ID3D12Device5> mDevice_v5;
	ComPtr<ID3D12CommandQueue> mCmdQueue_v0;
	ComPtr<ID3D12GraphicsCommandList4> mCmdList_v4;
	ComPtr<ID3D12CommandAllocator> mCmdAllocator_v0;
	BinaryFence mFence_v0;
	void initD3D12();

	ComPtr<ID3D12DescriptorHeap> mSrvUavHeap;
	UINT mSrvDescriptorSize;
	void createSrvUavHeap();

	ID3D12Resource* hdrTexture;
	ID3D12Resource* hdrUploadBuffer;
	ID3D12Resource* cacheTexture;
	ID3D12Resource* cacheUploadBuffer;
	ComPtr<ID3D12DescriptorHeap> mSamplerHeap;
	int mHdrResolution;
	float* calculateHdrCache(float* HDR, int width, int height);
	void loadHdrTexture();

	ComPtr<ID3D12RootSignature> mGlobalRS;
	ComPtr<ID3D12RootSignature> mHitGroupRS;
	ComPtr<ID3D12RootSignature> buildRootSignatures(const D3D12_ROOT_SIGNATURE_DESC& desc);
	void declareRootSignatures();

	dxShader mDxrLib;
	ComPtr<ID3D12StateObject> mRTPipeline;
	void buildRaytracingPipeline();

	GlobalConstants mGlobalConstants;
	ComPtr<ID3D12Resource> mGlobalConstantsBuffer;
	ComPtr<ID3D12Resource> mTracerOutBuffer;
	UINT64 mMaxBufferSize;
	ComPtr<ID3D12Resource> mReadBackBuffer;
	void initializeApplication();

	ComPtr<ID3D12Resource> mSceneObjectBuffer;
	ComPtr<ID3D12Resource> mVertexBuffer;
	ComPtr<ID3D12Resource> mIndexBuffer;
	ComPtr<ID3D12Resource> mMaterialBuffer;

	ComPtr<ID3D12Heap1> mShaderTableHeap_v1;
	ComPtr<ID3D12Resource> mRayGenShaderTable;
	ComPtr<ID3D12Resource> mMissShaderTable;
	ComPtr<ID3D12Resource> mHitGroupShaderTable;
	const WCHAR* cRayGenShaderName = L"rayGen";
	const WCHAR* cMissShaderName = L"missRay";
	const WCHAR* cMissShadowShaderName = L"missShadow";
	const WCHAR* cHitGroupName = L"hitGp";
	const WCHAR* cClosestHitshaderName = L"closestHit";
	vector<ObjectConstants> objConsts;
	void setupShaderTable();

	Scene* mScene;
	vector<ComPtr<ID3D12Resource>> mBottomLevelAccelerationStructure;
	ComPtr<ID3D12Resource> mTopLevelAccelerationStructure;
	vector<ComPtr<ID3D12Resource>> Scratch;
	ComPtr<ID3D12Resource> InstanceDesc;
	ComPtr<ID3D12Resource> createAS(
		const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& buildInputs,
		ComPtr<ID3D12Resource>* scrach);
	void buildBLAS(
		ComPtr<ID3D12Resource>* blas,
		ComPtr<ID3D12Resource>* scrach,
		const GPUMesh gpuMeshArr[],
		UINT numMeshes,
		UINT vertexStride,
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags);
	void buildTLAS(
		ComPtr<ID3D12Resource>* tlas,
		ComPtr<ID3D12Resource>* scrach,
		ComPtr<ID3D12Resource>* instanceDescArr,
		ID3D12Resource* const blasArr[],
		const dxTransform transformArr[],
		UINT numBlas,
		UINT instanceMultiplier,
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags);
	void buildAccelerationStructure();

	POINT mLastMousePos;
	Camera mCamera;

public:
	Camera getCamera() { return mCamera; }
	void onMouseDown(WPARAM btnState, int x, int y);
	void onMouseUp(WPARAM btnState, int x, int y);
	void onMouseMove(WPARAM btnState, int x, int y);

	void setupScene(const Scene* scene);
	TracedResult shootRays();

public:
	~DXRPathTracer();
	DXRPathTracer(HWND hwnd, UINT width, UINT height);
	void onSizeChanged(UINT width, UINT height);
	void update();
};