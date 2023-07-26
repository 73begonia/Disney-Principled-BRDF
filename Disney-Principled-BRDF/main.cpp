#include "D3D12Screen.h"
#include "DXRPathTracer.h"
#include "sceneLoader.h"
#include "timer.h"

HWND createWindow(const WCHAR* winTitle, UINT width, UINT height);

unique_ptr<D3D12Screen> g_screen;
unique_ptr<DXRPathTracer> g_tracer;

UINT g_width = 1280;
UINT g_height = 720;
bool g_minimized = false;

int main()
{
	HWND hwnd = createWindow(L"Disney-Principled-BRDF", g_width, g_height);
	ShowWindow(hwnd, SW_SHOW);

	g_tracer = make_unique<DXRPathTracer>(hwnd, g_width, g_height);
	g_screen = make_unique<D3D12Screen>(hwnd, g_width, g_height);

	SceneLoader sceneLoader;
	Scene* scene = sceneLoader.push_Teapot();
	g_tracer->setupScene(scene);

	double fps, old_fps = 0.0;
	while (IsWindow(hwnd))
	{
		if (!g_minimized)
		{
			g_tracer->update();
			TracedResult trResult = g_tracer->shootRays();
			g_screen->display(trResult);
		}

		MSG msg;
		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		fps = updateFPS(1.0);
		if (fps != old_fps)
		{
			printf("FPS: %f\n", fps);
			old_fps = fps;
		}
	}

	return 0;
}

LRESULT CALLBACK msgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

HWND createWindow(const WCHAR* winTitle, UINT width, UINT height)
{
	WNDCLASS wc = {};
	wc.lpfnWndProc = msgProc;
	wc.hInstance = GetModuleHandle(nullptr);
	wc.lpszClassName = L"Disney-Principled-BRDF";
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	RegisterClass(&wc);

	RECT r{ 0, 0, (LONG)width, (LONG)height };
	AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, false);

	HWND hWnd = CreateWindowW(
		L"Disney-Principled-BRDF",
		winTitle,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		r.right - r.left,
		r.bottom - r.top,
		nullptr,
		nullptr,
		wc.hInstance,
		nullptr);

	return hWnd;
}

LRESULT CALLBACK msgProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_LBUTTONDOWN:
		g_tracer->onMouseDown(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;
	case WM_LBUTTONUP:
		g_tracer->onMouseUp(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;
	case WM_MOUSEMOVE:
		g_tracer->onMouseMove(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;

	case WM_SIZE:
		if (g_screen)
		{
			UINT width = (UINT)LOWORD(lParam);
			UINT height = (UINT)HIWORD(lParam);
			if (width == 0 || height == 0)
			{
				g_minimized = true;
				return 0;
			}
			else if (g_minimized)
			{
				g_minimized = false;
			}

			g_tracer->onSizeChanged(width, height);
			g_screen->onSizeChanged(width, height);
		}
		return 0;
	}

	return DefWindowProc(hwnd, message, wParam, lParam);
}