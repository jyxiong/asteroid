#pragma once

class Screen
{
public:
    static void setWidth(int width);
    static int getWidth() { return s_width; }

    static void setHeight(int height);
    static int getHeight() { return s_height; }

    static void setSize(int width, int height);

    static float getAspect() { return s_aspect; }

private:
    static int s_width;
    static int s_height;
    static float s_aspect;
}; // class Screen
