#pragma once
#include "Scene.h"

class SceneLoader
{
	vector<Scene*> sceneArr;

	void initializeGeometryFromMeshes(Scene* scene, const vector<Mesh*>& meshes);
	void computeModelMatrices(Scene* scene);

public:
	Scene* getScene(uint sceneIdx) const { return sceneArr[sceneIdx]; }
	Scene* push_Teapot();
};