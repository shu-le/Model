import gradio as gr

mask_blur = gr.Slider(label="Mask Blur", minimum=0.1, maximum=7.0, value=5.0, step=0.01)