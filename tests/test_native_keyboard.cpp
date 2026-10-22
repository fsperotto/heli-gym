// Headless test of the real key callback, with GLFW calls replaced by stubs.
#include "py_api.h"
#include <cassert>
#include <cmath>
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
    Window::static_key_callback(nullptr, GLFW_KEY_M, 0, GLFW_PRESS, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_M, 0, GLFW_REPEAT, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_W, 0, GLFW_PRESS, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_W, 0, GLFW_REPEAT, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_T, 0, GLFW_PRESS, 0);
    Window::static_key_callback(nullptr, GLFW_KEY_T, 0, GLFW_REPEAT, 0);
    assert(window.show_info);
    Window::static_key_callback(nullptr, GLFW_KEY_H, 0, GLFW_PRESS, 0);
    assert(!window.show_info);
    Window::static_key_callback(nullptr, GLFW_KEY_H, 0, GLFW_REPEAT, 0);
    assert(!window.show_info);
    Window::static_key_callback(nullptr, GLFW_KEY_H, 0, GLFW_PRESS, 0);
    assert(window.show_info);
    assert(!window.follow_helicopter_bearing);
    Window::static_key_callback(nullptr, GLFW_KEY_V, 0, GLFW_PRESS, 0);
    assert(window.follow_helicopter_bearing);
    Window::static_key_callback(nullptr, GLFW_KEY_V, 0, GLFW_REPEAT, 0);
    assert(window.follow_helicopter_bearing);
    Window::static_key_callback(nullptr, GLFW_KEY_V, 0, GLFW_PRESS, 0);
    assert(!window.follow_helicopter_bearing);
    // A physical A key can be labelled Q on AZERTY.
    key_name = "q";
    Window::static_key_callback(nullptr, GLFW_KEY_A, 0, GLFW_PRESS, 0);
    assert(pop_key_press(&window) == GLFW_KEY_P);
    assert(pop_key_press(&window) == GLFW_KEY_UP);
    assert(pop_key_press(&window) == GLFW_KEY_R);
    assert(pop_key_press(&window) == GLFW_KEY_X);
    assert(pop_key_press(&window) == GLFW_KEY_M);
    assert(pop_key_press(&window) == GLFW_KEY_W);
    assert(pop_key_press(&window) == GLFW_KEY_T);
    assert(pop_key_press(&window) == GLFW_KEY_Q);
    assert(pop_key_press(&window) == -1);

    int section = window.create_guiText("Commands", ImVec2(), ImVec2());
    window.set_guiText_row_count(section, 7);
    assert(window.v_guiTextSection[section]->visible_rows == 7);
    window.set_guiText_row_count(section, -1);
    assert(window.v_guiTextSection[section]->visible_rows == -1);

    // GUI snapshots must outlive temporary ctypes-style input buffers.
    int target_section = window.create_guiText("Target", ImVec2(), ImVec2());
    {
        char label[] = "SPEED : %.2f ft/s";
        float speed = 0.0f;
        window.add_item_to_guiText(target_section, label, &speed);
        label[0] = 'X';
        speed = 0.31f;
        const guiText& row = window.v_guiTextSection[target_section]->textVector->at(0);
        assert(row.str == "SPEED : %.2f ft/s");
        assert(row.val == 0.0f);
    }
    {
        float values[] = {0.0f};
        window.set_guiText(target_section, values);
        values[0] = 123.0f;
    }
    assert(window.v_guiTextSection[target_section]->textVector->at(0).val == 0.0f);

    float short_values[] = {7.0f};
    assert(!set_guiText_checked(&window, target_section, 0, short_values));
    assert(!set_guiText_checked(&window, target_section, 2, short_values));
    assert(!set_guiText_checked(&window, -1, 1, short_values));
    assert(!set_guiText_checked(nullptr, target_section, 1, short_values));
    assert(window.v_guiTextSection[target_section]->textVector->at(0).val == 0.0f);
    assert(set_guiText_checked(&window, target_section, 1, short_values));
    short_values[0] = 999.0f;
    assert(window.v_guiTextSection[target_section]->textVector->at(0).val == 7.0f);
    assert(!add_guiText_checked(&window, -1, 0, nullptr, nullptr));
    assert(!add_guiText_checked(&window, target_section, 1, nullptr, short_values));

    assert(!set_guiText_row_count(&window, -1, 1));
    assert(!set_guiText_row_count(nullptr, target_section, 1));
    assert(set_guiText_row_count(&window, target_section, 1));
    set_fps(&window, 50.0f);
    set_fps(&window, 0.0f);
    set_fps(&window, -1.0f);
    set_fps(&window, std::nanf(""));
    assert(window.FPS_limit == 50.0f);
    set_fps(&window, 0.1f);
    assert(window.dt.count() > 9000000000LL);

    window.camera = new Camera(glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    window.update_helicopter_camera(0.0f, 0.0f, 0.0f, 0.0f);
    assert(std::abs(window.camera->Position.z - 30.0f) < 1e-5f);
    window.follow_helicopter_bearing = true;
    window.update_helicopter_camera(0.0f, 0.0f, 0.0f, 0.0f);
    assert(std::abs(window.camera->Position.x + 30.0f) < 1e-5f);
    assert(std::abs(window.camera->Position.y - 8.0f) < 1e-5f);
    window.update_helicopter_camera(0.0f, 0.0f, 0.0f, 1.57079632679f);
    assert(std::abs(window.camera->Position.z + 30.0f) < 1e-4f);
    Window::static_key_callback(nullptr, GLFW_KEY_ESCAPE, 0, GLFW_PRESS, 0);
    assert(close_requested);
    assert(pop_key_press(&window) == -1);
    std::cout << "Native keyboard and chase-camera checks passed.\n";
}
