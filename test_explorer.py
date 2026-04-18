import gradio as gr
def show_path(path):
    return f'Selected: {path}'
with gr.Blocks() as app:
    explorer = gr.FileExplorer(root_dir='.', file_count='single')
    out = gr.Textbox()
    explorer.change(show_path, inputs=explorer, outputs=out)
app.launch(server_name="0.0.0.0", server_port=7865)
