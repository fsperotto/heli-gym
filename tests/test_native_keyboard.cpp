// Headless test of the real key callback, with GLFW calls replaced by stubs.
#include "py_api.h"
#include <cassert>
#include <iostream>

static Window* owner;
static bool close_requested = false;
static const char* key_name = nullptr;
static double wait_timeout = 0;

extern "C" void* glfwGetWindowUserPointer(GLFWwindow*) { return owner; }
extern "C" const char* glfwGetKeyName(int, int) { return key_name; }
extern "C" void glfwSetWindowShouldClose(GLFWwindow*, int value) { close_requested = value; }
extern "C" void glfwWaitEventsTimeout(double timeout) { wait_timeout = timeout; }

int main()
{
    Window window;
    owner = &window;
    wait_events();
    assert(wait_timeout == 0.05);
    assert(pop_key_press(&window) == -1);
    Window::static_key_callback(nullptr, GLFW_KEY_P, 0, GLFW_PRESS, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_P, 0, GLFW_REPEAT, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_P, 0, GLFW_RELEASE, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_UP, 0, GLFW_PRESS, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_R, 0, GLFW_PRESS, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_R, 0, GLFW_REPEAT, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_X, 0, GLFW_PRESS, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_X, 0, GLFW_REPEAT, 0);
    // A physical A key can be labelled Q on AZERTY.
    key_name = "q";
    Window::static_key_callback(nullptr, GLFW_KEY_A, 0, GLFW_PRESS, 0);
    assert(pop_key_press(&window) == GLFW_KEY_P);
    assert(pop_key_press(&window) == GLFW_KEY_UP);
    assert(pop_key_press(&window) == GLFW_KEY_R);
    assert(pop_key_press(&window) == GLFW_KEY_X);
    assert(pop_key_press(&window) == GLFW_KEY_Q);
    assert(pop_key_press(&window) == -1);
    Window::static_key_callback(nullptr, GLFW_KEY_ESCAPE, 0, GLFW_PRESS, 0);
    assert(close_requested);
    assert(pop_key_press(&window) == -1);
    std::cout << "Native keyboard checks passed: press events, repeat filtering, ordering, layout, Escape.\n";
}
