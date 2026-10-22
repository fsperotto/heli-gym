#include "py_api.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

static std::vector<std::unique_ptr<Window>>& owned_windows()
{
    static std::vector<std::unique_ptr<Window>> windows;
    return windows;
}

static std::vector<std::unique_ptr<Model>>& owned_models()
{
    static std::vector<std::unique_ptr<Model>> models;
    return models;
}


Window* create_window(const unsigned int WIDTH,
		const unsigned int HEIGHT,
		const char* title)
{
    try {
        // GLFW termination and the ImGui context are global in this renderer.
        if (!owned_windows().empty()) throw std::runtime_error("Only one renderer window may be active; terminate it before creating another");
        std::unique_ptr<Window> window(new Window(WIDTH, HEIGHT, title));
        Window* pointer = window.get();
        owned_windows().push_back(std::move(window));
        return pointer;
    }
    catch (const std::exception& error) {
        std::cerr << "Renderer initialization failed: " << error.what() << std::endl;
        return nullptr;
    }
}


void close(Window* window)
{
	glfwSetWindowShouldClose(window->window, true);
}


bool is_close(Window* window)
{
	return glfwWindowShouldClose(window->window);
}


void render(Window* window)
{
	window->render();
}


void set_control_bars(Window* window, float collective, float longitudinal, float lateral, float pedals)
{
    const float values[4] = {collective, longitudinal, lateral, pedals};
    for (int i = 0; i < 4; ++i)
        window->control_values[i] = values[i];
    window->show_control_bars = true;
}

int pop_key_press(Window* window)
{
    if (window->key_presses.empty())
        return -1;
    const int key = window->key_presses.front();
    window->key_presses.pop_front();
    return key;
}

void wait_events()
{
    glfwWaitEventsTimeout(0.05);
}

void terminate()
{
    for (const auto& window : owned_windows()) window->shutdown_gui();
    glfwTerminate();
    owned_models().clear();
    owned_windows().clear();
}

Model* create_model(char* model_path, char* vertex_shader_file_path, char* fragment_shader_file_path)
{
    try {
        if (owned_windows().empty()) throw std::runtime_error("Create a renderer window before loading models");
        std::unique_ptr<Model> model(new Model(model_path, vertex_shader_file_path, fragment_shader_file_path));
        Model* pointer = model.get();
        owned_models().push_back(std::move(model));
        return pointer;
    }
    catch (const std::exception& error) {
        std::cerr << "Model loading failed: " << error.what() << std::endl;
        return nullptr;
    }
}


void add_permanent_to_window(Window* window, Model* model)
{
	window->add_permanent_drawables(model);
}


void add_instantaneous_to_window(Window* window, Model* model)
{
	window->add_instantaneous_drawables(model);
}


void translate_model(Model* model, float x, float y, float z)
{
	model->translate(glm::vec3(x, y, z));
}


void rotate_model(Model* model, float angle, float x, float y, float z)
{
	model->rotate(angle, glm::vec3(x, y, z));
}


void scale_model(Model* model, float x, float y, float z)
{
	model->scale(glm::vec3(x, y, z));
}


float get_fps(Window* window)
{
	return window->FPS;
}


void set_fps(Window* window, float fps)
{
    const double nanoseconds = 1e9 / static_cast<double>(fps);
    if (!std::isfinite(nanoseconds) || nanoseconds <= 0 ||
        nanoseconds >= static_cast<double>(std::numeric_limits<std::chrono::nanoseconds::rep>::max())) return;
    window->FPS_limit = fps;
    std::chrono::nanoseconds dt{static_cast<std::chrono::nanoseconds::rep>(nanoseconds)};
	window->dt = dt;
}


Camera* get_camera(Window* window)
{
	return window->camera;
}


void set_camera_pos(Camera* camera, float x, float y, float z)
{
	camera->Position = glm::vec3(x, y, z);
	camera->updateCameraVectors();
}


float* get_camera_pos(Camera* camera)
{
	return glm::value_ptr(camera->Position);
}



bool is_visible(Window* window)
{
	int visible = glfwGetWindowAttrib(window->window, GLFW_VISIBLE);
	return visible;
}


void hide_window(Window* window)
{
	glfwHideWindow(window->window);
}


void show_window(Window* window)
{
	glfwShowWindow(window->window);
}

int create_guiTextVector(Window* window, const char* title,
									float pos_x, float pos_y,
									float size_x, float size_y)
{
	ImVec2 pos = ImVec2(pos_x, pos_y);
	ImVec2 size = ImVec2(size_x, size_y);
	return window->create_guiText(title, pos, size);
}

void add_guiText(Window* window, int v_guiText_ind, int size, char** _str, float* _val)
{
	for (int i = 0; i < size; i++)
	{
		window->add_item_to_guiText(v_guiText_ind, _str[i], &_val[i] );
	}
}


void set_guiText(Window* window, int v_guiText_ind, float* _val)
{
	window->set_guiText(v_guiText_ind, _val);
}


void rotate_MR(Model* model, float phi, float theta, float psi)
{
	model->mainrotor = glm::vec3(phi, theta, psi);
}


void rotate_TR(Model* model, float phi, float theta, float psi)
{
	model->tailrotor = glm::vec3(phi, theta, psi);
}

void update_helicopter_camera(Window* window, float north, float east, float down, float bearing)
{
	window->update_helicopter_camera(north, east, down, bearing);
}

bool set_guiText_row_count(Window* window, int section, int count)
{
    if (!window || section < 0 || section >= window->v_guiTextSection.size()) return false;
    window->set_guiText_row_count(section, count);
    return true;
}

bool add_guiText_checked(Window* window, int section, int size, char** labels, float* values)
{
    if (!window || section < 0 || section >= window->v_guiTextSection.size() ||
        size < 0 || (size > 0 && (!labels || !values))) return false;
    for (int i = 0; i < size; ++i) if (!labels[i]) return false;
    add_guiText(window, section, size, labels, values);
    return true;
}

bool set_guiText_checked(Window* window, int section, int size, float* values)
{
    if (!window || section < 0 || section >= window->v_guiTextSection.size() ||
        size < 0 || size != window->v_guiTextSection[section]->textVector->size() ||
        (size > 0 && !values)) return false;
    set_guiText(window, section, values);
    return true;
}
