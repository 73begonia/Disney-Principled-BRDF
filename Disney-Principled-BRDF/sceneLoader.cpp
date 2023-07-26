#include "sceneLoader.h"

void SceneLoader::initializeGeometryFromMeshes(Scene* scene, const vector<Mesh*>& meshes)
{
	//scene->clear();

	uint numObjs = (uint)meshes.size();
	vector<Vertex>& vtxArr = scene->vtxArr;
	vector<Tridex>& tdxArr = scene->tdxArr;
	vector<SceneObject>& objArr = scene->objArr;

	objArr.resize(numObjs);

	uint totVertices = 0;
	uint totTridices = 0;

	for (uint i = 0; i < numObjs; ++i)
	{
		uint nowVertices = uint(meshes[i]->vtxArr.size());
		uint nowTridices = uint(meshes[i]->tdxArr.size());

		objArr[i].vertexOffset = totVertices;
		objArr[i].tridexOffset = totTridices;
		objArr[i].numVertices = nowVertices;
		objArr[i].numTridices = nowTridices;

		totVertices += nowVertices;
		totTridices += nowTridices;
	}

	vtxArr.resize(totVertices);
	tdxArr.resize(totTridices);

	for (uint i = 0; i < numObjs; ++i)
	{
		memcpy(&vtxArr[objArr[i].vertexOffset], &meshes[i]->vtxArr[0], sizeof(Vertex) * objArr[i].numVertices);
		memcpy(&tdxArr[objArr[i].tridexOffset], &meshes[i]->tdxArr[0], sizeof(Tridex) * objArr[i].numTridices);
	}
}

void SceneLoader::computeModelMatrices(Scene* scene)
{
	for (auto& obj : scene->objArr)
	{
		obj.modelMatrix = composeMatrix(obj.translation, obj.rotation, obj.scale);
	}
}

Scene* SceneLoader::push_Teapot()
{
	Scene* scene = new Scene;
	sceneArr.push_back(scene);

	Mesh ground = generateRectangleMesh(float3(0.0f, -2.f, 0.0f), float3(14.f, 0.f, 14.f), FaceDir::up);
	Mesh teapot = loadMeshFromOBJFile("../__models/teapot.obj", true);

	initializeGeometryFromMeshes(scene,
		{ &ground, &teapot });

	vector<Material>& mtlArr = scene->mtlArr;
	mtlArr.resize(2);

	//ground
	mtlArr[0].baseColor = 1.f;
	mtlArr[0].roughness = 0.1f;
	mtlArr[0].metallic = 0.f;
	
	//teapot
	mtlArr[1].baseColor = float3(1, 0.73, 0.25);
	mtlArr[1].roughness = 0.3f;
	mtlArr[1].metallic = 1.f;

	scene->objArr[0].materialIdx = 0;
	scene->objArr[1].materialIdx = 1;

	scene->objArr[0].translation = float3(0.0f);

	scene->objArr[1].translation = float3(0.f, -2.f, 0.f);
	scene->objArr[1].rotation = getRotationAsQuternion(float3(0, 1, 0), 90);
	scene->objArr[1].scale = 0.5f;

	computeModelMatrices(scene);

	return scene;
}