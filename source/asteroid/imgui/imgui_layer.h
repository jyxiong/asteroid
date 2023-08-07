#pragma once

#include "asteroid/base/layer.h"

namespace Asteroid
{

class ImGuiLayer : public Layer
{
public:
    ImGuiLayer();

    ~ImGuiLayer() override;

    void OnAttach() override;

    void OnDetach() override;

    void OnImGuiRender() override;

    void Begin();

    void End();

private:
    void ShowDockSpaceBegin();

    void ShowDockSpaceEnd();

private:
    float m_Time = 0.0f;

    static bool m_ShowDockSpace;
};

}