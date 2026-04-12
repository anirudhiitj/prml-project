// ─────────────────────────────────────────────────────────────────────────────
// gui.cpp — GTK3 Desktop GUI for Speech Separation
//
// Native C++ port of gui.py. Provides:
//   1. File picker for any audio/video format
//   2. ffmpeg conversion to 8kHz mono WAV
//   3. Conv-TasNet inference via the C++ binary
//   4. Playback and save-as for separated sources
//
// Build: cmake adds this automatically.
// Requires: libgtk-3-dev, ffmpeg
// ─────────────────────────────────────────────────────────────────────────────

#include <gtk/gtk.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>
#include <array>
#include <thread>
#include <functional>

namespace fs = std::filesystem;

// ── Configuration ────────────────────────────────────────────────────────────

static const int    SAMPLE_RATE = 8000;
static const char*  SUPPORTED_PATTERNS[] = {
    "*.mp4","*.mkv","*.avi","*.webm","*.mov","*.flac",
    "*.mp3","*.m4a","*.ogg","*.wav","*.aac","*.wma", nullptr
};

// ── Globals ──────────────────────────────────────────────────────────────────

struct AppState {
    // Paths
    std::string project_dir;
    std::string inference_bin;
    std::string checkpoint;
    std::string libtorch_lib;
    std::string output_dir;

    // Current operation
    std::string current_file;
    std::string source_paths[2];
    bool busy = false;

    // GTK widgets
    GtkWidget* window         = nullptr;
    GtkWidget* file_label     = nullptr;
    GtkWidget* sep_button     = nullptr;
    GtkWidget* status_label   = nullptr;
    GtkWidget* play_btn[2]    = {nullptr, nullptr};
    GtkWidget* save_btn[2]    = {nullptr, nullptr};
    GtkWidget* path_label[2]  = {nullptr, nullptr};
    GtkWidget* spinner        = nullptr;
};

static AppState g_app;

// ── Utility ──────────────────────────────────────────────────────────────────

static std::string get_project_dir() {
    // Resolve from binary location: build/gui → parent is project root
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        fs::path p(buf);
        return p.parent_path().parent_path().string(); // build/../
    }
    return ".";
}

static bool file_exists(const std::string& path) {
    return fs::exists(path);
}

static std::string find_checkpoint(const std::string& project_dir) {
    std::string best = project_dir + "/checkpoints/best_tasnet.pt";
    if (file_exists(best)) return best;
    std::string latest = project_dir + "/checkpoints/latest_tasnet.pt";
    if (file_exists(latest)) return latest;
    return "";
}

static bool command_exists(const char* cmd) {
    std::string check = std::string("which ") + cmd + " > /dev/null 2>&1";
    return system(check.c_str()) == 0;
}

// Run a command, capture stderr/stdout
static std::pair<int, std::string> run_cmd(const std::string& cmd) {
    std::string result;
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) return {-1, "Failed to run command"};
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    int rc = pclose(pipe);
    return {WEXITSTATUS(rc), result};
}

// ── Core Logic ───────────────────────────────────────────────────────────────

static bool convert_to_wav(const std::string& input, const std::string& output,
                           std::string& error) {
    std::string cmd = "ffmpeg -y -i \"" + input + "\" -vn -acodec pcm_s16le -ar "
                    + std::to_string(SAMPLE_RATE) + " -ac 1 \"" + output + "\"";
    auto [rc, out] = run_cmd(cmd);
    if (rc != 0) {
        error = out.size() > 500 ? out.substr(out.size() - 500) : out;
        return false;
    }
    return true;
}

static bool run_inference(const std::string& wav_path, const std::string& out_dir,
                          std::string& error) {
    std::string env = "LD_LIBRARY_PATH=\"" + g_app.libtorch_lib + ":$LD_LIBRARY_PATH\"";
    std::string cmd = env + " \"" + g_app.inference_bin + "\""
                    + " --model tasnet"
                    + " --checkpoint \"" + g_app.checkpoint + "\""
                    + " --input \"" + wav_path + "\""
                    + " --output_dir \"" + out_dir + "\""
                    + " --cpu";
    auto [rc, out] = run_cmd(cmd);
    if (rc != 0) {
        error = out.size() > 500 ? out.substr(out.size() - 500) : out;
        return false;
    }
    return true;
}

static void play_audio(const std::string& wav_path) {
    // Try common Linux audio players
    const char* players[] = {"paplay", "aplay", nullptr};
    for (int i = 0; players[i]; ++i) {
        if (command_exists(players[i])) {
            std::string cmd = std::string(players[i]) + " \"" + wav_path + "\" &";
            (void)system(cmd.c_str());
            return;
        }
    }
    // Fallback: ffplay
    if (command_exists("ffplay")) {
        std::string cmd = "ffplay -nodisp -autoexit \"" + wav_path + "\" &";
        (void)system(cmd.c_str());
    }
}

// ── GTK Helpers (thread-safe updates) ────────────────────────────────────────

struct UiUpdate {
    std::string text;
    std::string color;
    std::function<void()> fn;
};

static gboolean idle_set_status(gpointer data) {
    auto* u = static_cast<UiUpdate*>(data);
    // Escape text so chars like < > & don't break Pango markup
    gchar* escaped = g_markup_escape_text(u->text.c_str(), -1);
    std::string markup = "<span foreground=\"" + u->color + "\">" + escaped + "</span>";
    g_free(escaped);
    gtk_label_set_markup(GTK_LABEL(g_app.status_label), markup.c_str());
    delete u;
    return FALSE;
}

static gboolean idle_callback(gpointer data) {
    auto* fn = static_cast<std::function<void()>*>(data);
    (*fn)();
    delete fn;
    return FALSE;
}

static void set_status_async(const std::string& text, const std::string& color = "#f9e2af") {
    auto* u = new UiUpdate{text, color, {}};
    g_idle_add(idle_set_status, u);
}

static void run_on_main(std::function<void()> fn) {
    auto* fp = new std::function<void()>(std::move(fn));
    g_idle_add(idle_callback, fp);
}

// ── Separation Thread ────────────────────────────────────────────────────────

static void separation_thread() {
    fs::create_directories(g_app.output_dir);
    std::string tmp_wav = g_app.output_dir + "/_input_converted.wav";

    // Step 1: Convert
    set_status_async("⏳  Converting audio with ffmpeg …", "#f9e2af");
    std::string err;
    if (!convert_to_wav(g_app.current_file, tmp_wav, err)) {
        set_status_async("❌  ffmpeg error: " + err, "#f38ba8");
        run_on_main([]() {
            gtk_widget_set_sensitive(g_app.sep_button, TRUE);
            gtk_spinner_stop(GTK_SPINNER(g_app.spinner));
            g_app.busy = false;
        });
        return;
    }

    // Step 2: Inference
    set_status_async("⏳  Running Conv-TasNet inference …", "#f9e2af");
    if (!run_inference(tmp_wav, g_app.output_dir, err)) {
        set_status_async("❌  Inference error: " + err, "#f38ba8");
        run_on_main([]() {
            gtk_widget_set_sensitive(g_app.sep_button, TRUE);
            gtk_spinner_stop(GTK_SPINNER(g_app.spinner));
            g_app.busy = false;
        });
        return;
    }

    // Step 3: Check outputs
    g_app.source_paths[0] = g_app.output_dir + "/source_1.wav";
    g_app.source_paths[1] = g_app.output_dir + "/source_2.wav";

    if (file_exists(g_app.source_paths[0]) && file_exists(g_app.source_paths[1])) {
        set_status_async("✅  Separation complete!", "#a6e3a1");
        run_on_main([]() {
            for (int i = 0; i < 2; ++i) {
                gtk_label_set_text(GTK_LABEL(g_app.path_label[i]),
                                   g_app.source_paths[i].c_str());
                gtk_widget_set_sensitive(g_app.play_btn[i], TRUE);
                gtk_widget_set_sensitive(g_app.save_btn[i], TRUE);
            }
            gtk_widget_set_sensitive(g_app.sep_button, TRUE);
            gtk_spinner_stop(GTK_SPINNER(g_app.spinner));
            g_app.busy = false;
        });
    } else {
        set_status_async("❌  Output files not found", "#f38ba8");
        run_on_main([]() {
            gtk_widget_set_sensitive(g_app.sep_button, TRUE);
            gtk_spinner_stop(GTK_SPINNER(g_app.spinner));
            g_app.busy = false;
        });
    }
}

// ── Signal Handlers ──────────────────────────────────────────────────────────

static void on_file_select(GtkWidget*, gpointer) {
    GtkWidget* dialog = gtk_file_chooser_dialog_new(
        "Select Audio/Video File",
        GTK_WINDOW(g_app.window),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open",   GTK_RESPONSE_ACCEPT,
        nullptr);

    GtkFileFilter* filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Audio/Video files");
    for (int i = 0; SUPPORTED_PATTERNS[i]; ++i)
        gtk_file_filter_add_pattern(filter, SUPPORTED_PATTERNS[i]);
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

    GtkFileFilter* all = gtk_file_filter_new();
    gtk_file_filter_set_name(all, "All files");
    gtk_file_filter_add_pattern(all, "*");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), all);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        g_app.current_file = filename;
        std::string basename = fs::path(filename).filename().string();
        std::string display = "📄  " + basename;
        gtk_label_set_text(GTK_LABEL(g_app.file_label), display.c_str());
        gtk_widget_set_sensitive(g_app.sep_button, TRUE);
        gtk_label_set_text(GTK_LABEL(g_app.status_label), "");
        // Reset results
        for (int i = 0; i < 2; ++i) {
            gtk_label_set_text(GTK_LABEL(g_app.path_label[i]), "—");
            gtk_widget_set_sensitive(g_app.play_btn[i], FALSE);
            gtk_widget_set_sensitive(g_app.save_btn[i], FALSE);
        }
        g_free(filename);
    }
    gtk_widget_destroy(dialog);
}

static void on_separate(GtkWidget*, gpointer) {
    if (g_app.current_file.empty() || g_app.busy) return;
    g_app.busy = true;
    gtk_widget_set_sensitive(g_app.sep_button, FALSE);
    gtk_spinner_start(GTK_SPINNER(g_app.spinner));
    // Reset results
    for (int i = 0; i < 2; ++i) {
        gtk_label_set_text(GTK_LABEL(g_app.path_label[i]), "—");
        gtk_widget_set_sensitive(g_app.play_btn[i], FALSE);
        gtk_widget_set_sensitive(g_app.save_btn[i], FALSE);
    }
    std::thread(separation_thread).detach();
}

static void on_play(GtkWidget*, gpointer data) {
    int idx = GPOINTER_TO_INT(data);
    if (!g_app.source_paths[idx].empty())
        play_audio(g_app.source_paths[idx]);
}

static void on_save_as(GtkWidget*, gpointer data) {
    int idx = GPOINTER_TO_INT(data);
    if (g_app.source_paths[idx].empty()) return;

    GtkWidget* dialog = gtk_file_chooser_dialog_new(
        "Save Speaker Audio",
        GTK_WINDOW(g_app.window),
        GTK_FILE_CHOOSER_ACTION_SAVE,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Save",   GTK_RESPONSE_ACCEPT,
        nullptr);
    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);
    std::string default_name = "speaker_" + std::to_string(idx + 1) + ".wav";
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), default_name.c_str());

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char* dest = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        fs::copy_file(g_app.source_paths[idx], dest,
                      fs::copy_options::overwrite_existing);
        std::string msg = "💾  Saved to " + std::string(dest);
        gtk_label_set_markup(GTK_LABEL(g_app.status_label),
            ("<span foreground=\"#a6e3a1\">" + msg + "</span>").c_str());
        g_free(dest);
    }
    gtk_widget_destroy(dialog);
}

// ── CSS Theme ────────────────────────────────────────────────────────────────

static void apply_css() {
    const char* css_data = R"CSS(
        window {
            background-color: #1e1e2e;
        }
        .title-label {
            color: #89b4fa;
            font-weight: bold;
            font-size: 20px;
        }
        .subtitle-label {
            color: #6c7086;
            font-size: 11px;
        }
        .card {
            background-color: #313244;
            border-radius: 8px;
            padding: 12px;
        }
        .file-label {
            color: #a6adc8;
            font-size: 12px;
        }
        .sep-button {
            background: linear-gradient(135deg, #89b4fa, #74c7ec);
            color: #1e1e2e;
            font-weight: bold;
            font-size: 14px;
            border-radius: 6px;
            padding: 8px 24px;
            border: none;
        }
        .sep-button:hover {
            background: linear-gradient(135deg, #74c7ec, #89dceb);
        }
        .sep-button:disabled {
            opacity: 0.5;
        }
        .status-label {
            font-size: 11px;
        }
        .speaker-label {
            color: #a6e3a1;
            font-weight: bold;
            font-size: 12px;
        }
        .path-label {
            color: #6c7086;
            font-size: 10px;
        }
        .action-button {
            background-color: #45475a;
            color: #cdd6f4;
            border-radius: 4px;
            padding: 4px 10px;
            border: none;
            font-size: 10px;
        }
        .action-button:hover {
            background-color: #585b70;
        }
        .action-button:disabled {
            opacity: 0.4;
        }
        .footer-label {
            color: #585b70;
            font-size: 9px;
        }
        .select-button {
            background-color: #313244;
            color: #a6adc8;
            border: 2px dashed #585b70;
            border-radius: 8px;
            padding: 24px 16px;
            font-size: 12px;
        }
        .select-button:hover {
            border-color: #89b4fa;
            color: #cdd6f4;
        }
    )CSS";

    GtkCssProvider* provider = gtk_css_provider_new();
    gtk_css_provider_load_from_data(provider, css_data, -1, nullptr);
    gtk_style_context_add_provider_for_screen(
        gdk_screen_get_default(),
        GTK_STYLE_PROVIDER(provider),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(provider);
}

// ── Build UI ─────────────────────────────────────────────────────────────────

static void build_ui(GtkApplication* app) {
    g_app.window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(g_app.window), "Speech Separation");
    gtk_window_set_default_size(GTK_WINDOW(g_app.window), 620, 480);
    gtk_window_set_resizable(GTK_WINDOW(g_app.window), FALSE);

    apply_css();

    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_set_border_width(GTK_CONTAINER(vbox), 20);
    gtk_container_add(GTK_CONTAINER(g_app.window), vbox);

    // ── Title ────────────────────────────────────────────────────────────
    GtkWidget* title = gtk_label_new(nullptr);
    gtk_label_set_markup(GTK_LABEL(title),
        "<span font='20' weight='bold' foreground='#89b4fa'>🎙  Speech Separation</span>");
    gtk_box_pack_start(GTK_BOX(vbox), title, FALSE, FALSE, 4);

    GtkWidget* subtitle = gtk_label_new("Drop an MP4 / audio file → 2 separated speakers");
    GtkStyleContext* sub_ctx = gtk_widget_get_style_context(subtitle);
    gtk_style_context_add_class(sub_ctx, "subtitle-label");
    gtk_box_pack_start(GTK_BOX(vbox), subtitle, FALSE, FALSE, 0);

    // ── File selector ────────────────────────────────────────────────────
    GtkWidget* file_btn = gtk_button_new();
    g_app.file_label = gtk_label_new("📂  Click here to select a file\n\nSupports: MP4, MKV, AVI, MP3, WAV, FLAC …");
    gtk_label_set_justify(GTK_LABEL(g_app.file_label), GTK_JUSTIFY_CENTER);
    gtk_container_add(GTK_CONTAINER(file_btn), g_app.file_label);
    GtkStyleContext* fb_ctx = gtk_widget_get_style_context(file_btn);
    gtk_style_context_add_class(fb_ctx, "select-button");
    gtk_box_pack_start(GTK_BOX(vbox), file_btn, FALSE, FALSE, 16);
    g_signal_connect(file_btn, "clicked", G_CALLBACK(on_file_select), nullptr);

    // ── Separate button + spinner ────────────────────────────────────────
    GtkWidget* sep_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_halign(sep_hbox, GTK_ALIGN_CENTER);

    g_app.sep_button = gtk_button_new_with_label("✦  Separate Speakers");
    GtkStyleContext* sb_ctx = gtk_widget_get_style_context(g_app.sep_button);
    gtk_style_context_add_class(sb_ctx, "sep-button");
    gtk_widget_set_sensitive(g_app.sep_button, FALSE);
    g_signal_connect(g_app.sep_button, "clicked", G_CALLBACK(on_separate), nullptr);
    gtk_box_pack_start(GTK_BOX(sep_hbox), g_app.sep_button, FALSE, FALSE, 0);

    g_app.spinner = gtk_spinner_new();
    gtk_box_pack_start(GTK_BOX(sep_hbox), g_app.spinner, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(vbox), sep_hbox, FALSE, FALSE, 8);

    // ── Status label ─────────────────────────────────────────────────────
    g_app.status_label = gtk_label_new("");
    gtk_label_set_line_wrap(GTK_LABEL(g_app.status_label), TRUE);
    gtk_label_set_max_width_chars(GTK_LABEL(g_app.status_label), 70);
    GtkStyleContext* sl_ctx = gtk_widget_get_style_context(g_app.status_label);
    gtk_style_context_add_class(sl_ctx, "status-label");
    gtk_box_pack_start(GTK_BOX(vbox), g_app.status_label, FALSE, FALSE, 4);

    // ── Speaker result cards ─────────────────────────────────────────────
    for (int i = 0; i < 2; ++i) {
        GtkWidget* card = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
        GtkStyleContext* c_ctx = gtk_widget_get_style_context(card);
        gtk_style_context_add_class(c_ctx, "card");

        std::string spk_text = "<span foreground='#a6e3a1' weight='bold'>🔊 Speaker "
                             + std::to_string(i + 1) + "</span>";
        GtkWidget* spk_label = gtk_label_new(nullptr);
        gtk_label_set_markup(GTK_LABEL(spk_label), spk_text.c_str());
        gtk_box_pack_start(GTK_BOX(card), spk_label, FALSE, FALSE, 4);

        g_app.path_label[i] = gtk_label_new("—");
        GtkStyleContext* pl_ctx = gtk_widget_get_style_context(g_app.path_label[i]);
        gtk_style_context_add_class(pl_ctx, "path-label");
        gtk_label_set_ellipsize(GTK_LABEL(g_app.path_label[i]), PANGO_ELLIPSIZE_MIDDLE);
        gtk_label_set_max_width_chars(GTK_LABEL(g_app.path_label[i]), 40);
        gtk_box_pack_start(GTK_BOX(card), g_app.path_label[i], TRUE, TRUE, 0);

        g_app.save_btn[i] = gtk_button_new_with_label("💾 Save");
        GtkStyleContext* sv_ctx = gtk_widget_get_style_context(g_app.save_btn[i]);
        gtk_style_context_add_class(sv_ctx, "action-button");
        gtk_widget_set_sensitive(g_app.save_btn[i], FALSE);
        g_signal_connect(g_app.save_btn[i], "clicked",
                         G_CALLBACK(on_save_as), GINT_TO_POINTER(i));
        gtk_box_pack_end(GTK_BOX(card), g_app.save_btn[i], FALSE, FALSE, 0);

        g_app.play_btn[i] = gtk_button_new_with_label("▶ Play");
        GtkStyleContext* pb_ctx = gtk_widget_get_style_context(g_app.play_btn[i]);
        gtk_style_context_add_class(pb_ctx, "action-button");
        gtk_widget_set_sensitive(g_app.play_btn[i], FALSE);
        g_signal_connect(g_app.play_btn[i], "clicked",
                         G_CALLBACK(on_play), GINT_TO_POINTER(i));
        gtk_box_pack_end(GTK_BOX(card), g_app.play_btn[i], FALSE, FALSE, 0);

        gtk_box_pack_start(GTK_BOX(vbox), card, FALSE, FALSE, 4);
    }

    // ── Footer ───────────────────────────────────────────────────────────
    std::string ckpt_name = g_app.checkpoint.empty() ? "none"
                          : fs::path(g_app.checkpoint).filename().string();
    std::string footer_text = "Model: Conv-TasNet (8.2M) · 2 speakers · 8 kHz · Checkpoint: "
                            + ckpt_name;
    GtkWidget* footer = gtk_label_new(footer_text.c_str());
    GtkStyleContext* ft_ctx = gtk_widget_get_style_context(footer);
    gtk_style_context_add_class(ft_ctx, "footer-label");
    gtk_box_pack_end(GTK_BOX(vbox), footer, FALSE, FALSE, 4);

    gtk_widget_show_all(g_app.window);
}

// ── Activation ───────────────────────────────────────────────────────────────

static void activate(GtkApplication* app, gpointer) {
    // Resolve paths
    g_app.project_dir  = get_project_dir();
    g_app.inference_bin = g_app.project_dir + "/build/inference";
    g_app.libtorch_lib  = g_app.project_dir + "/libtorch/lib";
    g_app.output_dir    = g_app.project_dir + "/output";
    g_app.checkpoint    = find_checkpoint(g_app.project_dir);

    // Check dependencies
    std::string issues;
    if (!file_exists(g_app.inference_bin))
        issues += "• Inference binary not found at:\n  " + g_app.inference_bin
                + "\n  → Run: cd build && cmake .. && make -j$(nproc)\n\n";
    if (g_app.checkpoint.empty())
        issues += "• No checkpoint found in checkpoints/\n  → Train the model first\n\n";
    if (!command_exists("ffmpeg"))
        issues += "• ffmpeg not found\n  → sudo apt install ffmpeg\n\n";

    if (!issues.empty()) {
        GtkWidget* dlg = gtk_message_dialog_new(
            nullptr, GTK_DIALOG_MODAL, GTK_MESSAGE_ERROR, GTK_BUTTONS_OK,
            "Setup Issues:\n\n%s", issues.c_str());
        gtk_window_set_title(GTK_WINDOW(dlg), "Speech Separation — Error");
        gtk_dialog_run(GTK_DIALOG(dlg));
        gtk_widget_destroy(dlg);
        return;
    }

    build_ui(app);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    GtkApplication* app = gtk_application_new(
        "org.prml.speechseparation",
        G_APPLICATION_FLAGS_NONE);
    g_signal_connect(app, "activate", G_CALLBACK(activate), nullptr);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    return status;
}
