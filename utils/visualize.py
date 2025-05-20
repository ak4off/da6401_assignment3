import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np
import imageio
import os
from IPython.display import display, HTML, clear_output
import time


def create_interactive_connectivity111(attn_matrix, input_seq, output_seq, filename="attention.html"):
    """
    Creates a complete interactive connectivity visualization with:
    - Output characters on top
    - Input characters below
    - Green connection lines
    - Hover/click interaction
    - Threshold slider
    """
    
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Attention Connectivity Visualization</title>
    <style>
        body { 
            font-family: 'Arial Unicode MS', 'Noto Sans Devanagari', Arial, sans-serif;
            margin: 20px; 
            text-align: center;
        }
        .container { 
            display: inline-block; 
            text-align: center;
            margin: 0 auto;
        }
        .output-chars { 
            display: flex; 
            justify-content: center;
            margin-bottom: 40px;
        }
        .input-chars { 
            display: flex; 
            justify-content: center;
            margin-top: 20px;
        }
        .char { 
            padding: 10px 15px;
            margin: 5px;
            font-size: 24px;
            position: relative;
            cursor: pointer;
            transition: all 0.3s;
            min-width: 30px;
            text-align: center;
        }
        .output-char { 
            background-color: #f0f0f0; 
            border-radius: 5px; 
        }
        .input-char { 
            background-color: #e0e0e0; 
            border-radius: 3px; 
        }
        .connection-line {
            position: absolute;
            background-color: rgba(0, 200, 0, 0.5);
            height: 4px;
            transform-origin: left center;
            z-index: -1;
            pointer-events: none;
        }
        .selected { 
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
        }
        .highlighted { 
            background-color: rgba(76, 175, 80, 0.3);
            transform: scale(1.1);
        }
        .controls { 
            margin: 20px 0; 
        }
        .slider { 
            width: 300px; 
            margin: 0 10px; 
        }
        .threshold-value { 
            display: inline-block; 
            width: 50px; 
        }
        h2 { color: #333; }
        h5 { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Attention Connectivity Visualization</h2>
        <h5>[Hover over the devanagri characters]</h5>
        <div class="controls">
            <label>Connection Threshold: </label>
            <input type="range" min="0" max="100" value="30" class="slider" id="thresholdSlider">
            <span class="threshold-value" id="thresholdValue">0.30</span>
        </div>
        
        <div class="output-chars" id="outputChars"></div>
        <div class="input-chars" id="inputChars"></div>
    </div>

    <script>
        // Convert Python data to JS format
        const attentionData = {attn_matrix};
        const inputChars = {input_seq};
        const outputChars = {output_seq};
        
        let currentSelected = 0;
        let threshold = 0.3;
        
        function initVisualization() {
            renderOutputChars();
            renderInputChars();
            updateConnections();
            
            // Setup threshold slider
            document.getElementById('thresholdSlider').addEventListener('input', function(e) {
                threshold = parseInt(e.target.value) / 100;
                document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
                updateConnections();
            });
            
            // Handle window resize
            window.addEventListener('resize', updateConnections);
        }
        
        function renderOutputChars() {
            const container = document.getElementById('outputChars');
            container.innerHTML = '';
            
            outputChars.forEach((char, idx) => {
                const charElement = document.createElement('div');
                charElement.className = `char output-char ${idx === currentSelected ? 'selected' : ''}`;
                charElement.textContent = char;
                charElement.dataset.index = idx;
                
                charElement.addEventListener('mouseover', () => selectCharacter(idx));
                charElement.addEventListener('click', () => selectCharacter(idx));
                
                container.appendChild(charElement);
            });
        }
        
        function renderInputChars() {
            const container = document.getElementById('inputChars');
            container.innerHTML = '';
            
            inputChars.forEach((char, idx) => {
                const charElement = document.createElement('div');
                charElement.className = 'char input-char';
                charElement.textContent = char;
                charElement.dataset.index = idx;
                container.appendChild(charElement);
            });
        }
        
        function selectCharacter(idx) {
            currentSelected = idx;
            renderOutputChars();
            updateConnections();
        }
        
        function updateConnections() {
            // Clear existing connections
            document.querySelectorAll('.connection-line').forEach(el => el.remove());
            document.querySelectorAll('.input-char').forEach(el => el.classList.remove('highlighted'));
            
            const outputChar = document.querySelector(`.output-char[data-index="${currentSelected}"]`);
            if (!outputChar) return;
            
            const outputRect = outputChar.getBoundingClientRect();
            const attentionWeights = attentionData[currentSelected];
            const maxWeight = Math.max(...attentionWeights);
            
            inputChars.forEach((_, idx) => {
                const inputChar = document.querySelector(`.input-char[data-index="${idx}"]`);
                if (!inputChar) return;
                
                const inputRect = inputChar.getBoundingClientRect();
                const normalizedWeight = attentionWeights[idx] / maxWeight;
                
                if (normalizedWeight >= threshold) {
                    inputChar.classList.add('highlighted');
                    
                    const line = document.createElement('div');
                    line.className = 'connection-line';
                    
                    const startX = outputRect.left + outputRect.width/2 - window.scrollX;
                    const startY = outputRect.top + outputRect.height - window.scrollY;
                    const endX = inputRect.left + inputRect.width/2 - window.scrollX;
                    const endY = inputRect.top - window.scrollY;
                    
                    const length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
                    const angle = Math.atan2(endY - startY, endX - startX) * 180 / Math.PI;
                    
                    line.style.width = `${length}px`;
                    line.style.left = `${startX}px`;
                    line.style.top = `${startY}px`;
                    line.style.transform = `rotate(${angle}deg)`;
                    line.style.opacity = normalizedWeight;
                    
                    document.body.appendChild(line);
                }
            });
        }
        
        // Initialize visualization
        document.addEventListener('DOMContentLoaded', initVisualization);
    </script>
</body>
</html>"""
    
    # Prepare data for JavaScript
    attn_matrix_json = [[float(w) for w in row] for row in attn_matrix]
    input_seq_json = [str(c) for c in input_seq]
    output_seq_json = [str(c) for c in output_seq]
    
    # Insert data into template
    import json
    html_content = html_template \
        .replace('{attn_matrix}', json.dumps(attn_matrix_json)) \
        .replace('{input_seq}', json.dumps(input_seq_json)) \
        .replace('{output_seq}', json.dumps(output_seq_json))
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # print(f"Visualization saved to {filename}")
    return filename

def get_clr(value):
    colors = ['#FFFFFF'] * 3 + \
             ['#f9e8e8'] * 2 + ['#f9d4d4'] + ['#f9bdbd'] + ['#f8a8a8'] + \
             ['#f68f8f'] * 2 + ['#f47676'] + ['#f45f5f'] * 2 + ['#f34343'] * 2 + \
             ['#f33b3b'] * 3 + ['#f42e2e'] * 2
    idx = min(int(value * 20), len(colors) - 1)
    return colors[idx]

def cstr(word, color='black'):
    if word == ' ':
        return f"<text style='color:#000;padding-left:10px;background-color:{color}'> </text>"
    else:
        return f"<text style='color:#000;background-color:{color}'>{word} </text>"

def print_color(tuples):
    html = ''.join([cstr(token, color) for token, color in tuples])
    display(HTML(html))

def visualize_connectivity(attn_matrix, input_seq, output_seq, delay=0.5):
    for i, output_char in enumerate(output_seq):
        weights = attn_matrix[i]
        token_attention = [(char, get_clr(weights[j])) for j, char in enumerate(input_seq)]
        decoder_step = [(output_char, "#aaffaa")] + [(" ", "#FFFFFF")] * (len(input_seq)-1)

        print_color(token_attention)
        print_color(decoder_step)
        time.sleep(delay)
        if i < len(output_seq) - 1:
            clear_output(wait=True)

def draw_connectivity(attn_matrix, input_seq, output_seq, idx=0):
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_matrix.numpy(), xticklabels=input_seq, yticklabels=output_seq, cmap="viridis")
    plt.xlabel("Input Sequence")
    plt.ylabel("Output Sequence")
    plt.title(f"Attention Heatmap {idx}")
    plt.tight_layout()
    plt.show()

def attention_to_html(attn_matrix, input_seq, output_seq):
    html = ""
    for i, output_char in enumerate(output_seq):
        weights = attn_matrix[i]
        token_attention = [(char, get_clr(weights[j])) for j, char in enumerate(input_seq)]
        decoder_step = [(output_char, "#aaffaa")] + [(" ", "#FFFFFF")] * (len(input_seq)-1)
        html += ''.join([cstr(token, color) for token, color in token_attention]) + "<br/>"
        html += ''.join([cstr(token, color) for token, color in decoder_step]) + "<br/><br/>"
    return html

# def save_attention_gif(attn_matrix, input_seq, output_seq, out_path="attn.gif"):
#     images = []
#     for i, output_char in enumerate(output_seq):
#         fig = plt.figure()
#         sns.heatmap(attn_matrix[i:i+1], xticklabels=input_seq, yticklabels=[output_char], cmap="Reds", cbar=False)
#         plt.title(f"Decoder Step {i}: {output_char}")
#         plt.xlabel("Input")
#         plt.tight_layout()
        
#         temp_path = f"temp_frame_{i}.png"
#         plt.savefig(temp_path)
#         plt.close(fig)
#         images.append(imageio.imread(temp_path))
#         os.remove(temp_path)

#     imageio.mimsave(out_path, images, duration=0.8)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

def log_attention_to_wandb(attn_matrix, input_seq, output_seq, idx=0, table=None):
    # Use the Lohit Devanagari font for all text in the plot
    devanagari_font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf")

    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(attn_matrix.numpy(), 
                     xticklabels=input_seq, 
                     yticklabels=output_seq, 
                     cmap="viridis",
                     cbar=False)

    # Apply the font to tick labels
    ax.set_xticklabels(input_seq, fontproperties=devanagari_font, rotation=90)
    ax.set_yticklabels(output_seq, fontproperties=devanagari_font, rotation=0)

    # Titles and labels
    ax.set_title(f"Sample {idx}: {''.join(input_seq)} → {''.join(output_seq)}", fontproperties=devanagari_font, fontsize=12)
    ax.set_xlabel("Input", fontproperties=devanagari_font, fontsize=10)
    ax.set_ylabel("Output", fontproperties=devanagari_font, fontsize=10)

    plt.tight_layout()

    if table:
        table.add_data(''.join(input_seq), ''.join(output_seq), wandb.Image(fig))

    plt.close(fig)

def log_attention_to_wandb1(attn_matrix, input_seq, output_seq, idx=0, table=None):
    # 1. Plot heatmap
    plt.rcParams['font.family'] = 'DejaVu Sans' 
    fig = plt.figure()
    sns.heatmap(attn_matrix.numpy(), xticklabels=input_seq, yticklabels=output_seq, cmap="viridis")
    plt.title(f"Sample {idx}: {''.join(input_seq)} → {''.join(output_seq)}")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.tight_layout()

    if table:
        table.add_data(''.join(input_seq), ''.join(output_seq), wandb.Image(fig))
    plt.close(fig)

    # ❌ Removed GIF generation and logging
    # gif_path = f"attn_{idx}.gif"
    # save_attention_gif(attn_matrix, input_seq, output_seq, out_path=gif_path)
    # wandb.log({f"attention_gif_{idx}": wandb.Video(gif_path, fps=1, format="gif")})
    # os.remove(gif_path)

    # 2. Log HTML version
    html_str = attention_to_html(attn_matrix, input_seq, output_seq)
    wandb.log({f"attention_html_{idx}": wandb.Html(html_str)})
    # wandb.log({f"connectivity_{idx}": wandb.Html(html_str)})



# def get_clr(value):
#     """Returns a hex color based on intensity value (0 to 1 scaled to 0 to 20)"""
#     colors = ['#FFFFFF'] * 3 + \
#              ['#f9e8e8'] * 2 + ['#f9d4d4'] + ['#f9bdbd'] + ['#f8a8a8'] + \
#              ['#f68f8f'] * 2 + ['#f47676'] + ['#f45f5f'] * 2 + ['#f34343'] * 2 + \
#              ['#f33b3b'] * 3 + ['#f42e2e'] * 2
#     idx = min(int(value * 20), len(colors) - 1)
#     return colors[idx]

# def cstr(word, color='black'):
#     if word == ' ':
#         return f"<text style='color:#000;padding-left:10px;background-color:{color}'> </text>"
#     else:
#         return f"<text style='color:#000;background-color:{color}'>{word} </text>"

# def print_color(tuples):
#     """Prints tokens with color based on attention weight"""
#     html = ''.join([cstr(token, color) for token, color in tuples])
#     display(HTML(html))

# def visualize_connectivity(attn_matrix, input_seq, output_seq, delay=0.5):
#     """
#     Shows attention from each decoder output step to input sequence with color mapping.
#     attn_matrix: (output_len, input_len)
#     input_seq: list of input characters
#     output_seq: list of predicted output characters
#     """
#     import time
#     from IPython.display import clear_output

#     for i, output_char in enumerate(output_seq):
#         weights = attn_matrix[i]
#         token_attention = [(char, get_clr(weights[j])) for j, char in enumerate(input_seq)]
#         decoder_step = [(output_char, "#aaffaa")] + [(" ", "#FFFFFF")] * (len(input_seq)-1)

#         print_color(token_attention)
#         print_color(decoder_step)
#         time.sleep(delay)
#         if i < len(output_seq) - 1:
#             clear_output(wait=True)


def plot_attention_heatmap_errr(attention, input_tokens, output_tokens, idx=0, save_path=None):
    """
    attention: Tensor of shape (target_len, source_len)
    input_tokens: List[str] of input tokens (e.g., Latin chars)
    output_tokens: List[str] of output tokens (e.g., Native chars)
    idx: Index for title labeling
    save_path: Optional path to save the figure
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(attention.cpu().detach().numpy(), xticklabels=input_tokens, yticklabels=output_tokens, cmap="viridis")
    plt.xlabel("Input Sequence")
    plt.ylabel("Output Sequence")
    plt.title(f"Attention Heatmap {idx}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
def plot_attention_heatmap(attention, input_tokens, output_tokens, idx=0, save_path=None):
    """
    attention: Tensor of shape (target_len, source_len) or (batch_size, target_len, source_len)
    input_tokens: List[str] of input tokens (e.g., Latin chars)
    output_tokens: List[str] of output tokens (e.g., Native chars)
    idx: Index for title labeling
    save_path: Optional path to save the figure
    """
    # Check if attention has more than 2 dimensions (e.g., batch size or multi-head attention)
    if len(attention.shape) > 2:
        # For simplicity, let's visualize the first element in the batch
        attention = attention[0]  # Select the first batch (adjust as needed for your use case)

    plt.figure(figsize=(6, 5))
    sns.heatmap(attention.cpu().detach().numpy(), xticklabels=input_tokens, yticklabels=output_tokens, cmap="viridis")
    plt.xlabel("Input Sequence")
    plt.ylabel("Output Sequence")
    plt.title(f"Attention Heatmap {idx}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# for wandb logging
def plot_grid_heatmaps(attention_list, input_list, output_list):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import wandb
    from matplotlib import font_manager

    # Load Devanagari font
    devanagari_font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf")

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i, ax in enumerate(axes.flat):
        if i >= len(attention_list):
            break

        if attention_list[i].shape != (len(output_list[i]), len(input_list[i])):
            print(f"[Warning] Shape mismatch in heatmap: attention {attention_list[i].shape}, y {len(output_list[i])}, x {len(input_list[i])}")
            continue

        sns.heatmap(attention_list[i].cpu().detach().numpy(),
                    ax=ax,
                    xticklabels=input_list[i],
                    yticklabels=output_list[i],
                    cmap="viridis")

        # Apply font to tick labels
        ax.set_xticklabels(input_list[i], fontproperties=devanagari_font, rotation=90)
        ax.set_yticklabels(output_list[i], fontproperties=devanagari_font, rotation=0)

        ax.set_title(f"Sample {i}", fontproperties=devanagari_font, fontsize=12)
        ax.set_xlabel("Input", fontproperties=devanagari_font, fontsize=10)
        ax.set_ylabel("Output", fontproperties=devanagari_font, fontsize=10)

    plt.tight_layout()
    wandb.log({"attention_grid": wandb.Image(fig, caption="Attention Heatmaps")})
    plt.close(fig)

# def plot_grid_heatmaps_working(attention_list, input_list, output_list):
#     fig, axes = plt.subplots(3, 3, figsize=(18, 12))
#     for i, ax in enumerate(axes.flat):
#         if i >= len(attention_list):
#             break
        
#         attn = attention_list[i]
#         # If attn is 3D, pick the first item in the batch
#         if attn.dim() == 3:
#             attn = attn[0]

#         sns.heatmap(attn.cpu().detach().numpy(), ax=ax,
#                     xticklabels=input_list[i], yticklabels=output_list[i],
#                     cmap="viridis")
#         ax.set_title(f"Sample {i}")
#         ax.set_xlabel("Input")
#         ax.set_ylabel("Output")
#     plt.tight_layout()
#     plt.savefig("attention_grid.png")

def plot_grid_heatmaps_err(attention_list, input_list, output_list):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i, ax in enumerate(axes.flat):
        if i >= len(attention_list):
            break
        sns.heatmap(attention_list[i].cpu().detach().numpy(), ax=ax, xticklabels=input_list[i], yticklabels=output_list[i], cmap="viridis")
        ax.set_title(f"Sample {i}")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
    plt.tight_layout()
    plt.savefig("attention_grid.png")

import matplotlib.pyplot as plt
import networkx as nx
import wandb

def draw_connectivity(attention, input_seq, output_seq, idx=None, save_path=None, log_to_wandb=True):
    """
    Draws a connectivity graph using attention weights between input_seq and output_seq.

    attention: shape (tgt_len, src_len)
    input_seq: list of input tokens (e.g. Latin)
    output_seq: list of output tokens (e.g. Devanagari)
    idx: optional identifier for title or saving
    save_path: if provided, saves the plot as a PNG
    log_to_wandb: if True, logs the figure to wandb
    """
    G = nx.DiGraph()
    threshold = 0.05  # Minimum attention to draw edge

    for i, out_char in enumerate(output_seq):
        for j, in_char in enumerate(input_seq):
            weight = attention[i][j]
            if weight > threshold:
                G.add_edge(f"{in_char}_{j}", f"{out_char}_{i}", weight=weight)

    pos = nx.spring_layout(G, seed=42)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, edge_color=[float(w) for w in edge_weights], edge_cmap=plt.cm.Reds,
            node_color='lightblue', font_size=10, node_size=2000, ax=ax)

    title = f"Connectivity Graph {idx}" if idx is not None else "Connectivity Graph"
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    if log_to_wandb:
        wandb.log({f"connectivity_graph_{idx if idx else 'sample'}": wandb.Image(fig)})
    plt.close()
import imageio
from PIL import Image, ImageDraw, ImageFont

# Update font loading in save_attention_gif
font_path = "/usr/share/fonts/truetype/samyak/Samyak-Devanagari.ttf"  # change path as per your system
font = ImageFont.truetype(font_path, size=16)
try:
    font = ImageFont.truetype("Samyak-Devanagari.ttf", size=16)
except:
    print("Warning: Devanagari font not found. Falling back to default.")
    font = ImageFont.load_default()

# def save_attention_gif(attn_matrix, input_seq, output_seq, save_path="attention.gif", delay=500,out_path="attn.gif"):
#     """
#     Save attention visualization as an animated GIF.
#     attn_matrix: (output_len, input_len)
#     input_seq: list of input characters
#     output_seq: list of predicted output characters
#     """
#     images = []
#     frames = []
#     font = ImageFont.load_default()

#     for i, output_char in enumerate(output_seq):
#         weights = attn_matrix[i]
#         weights = (weights / weights.max()).tolist()
#         width = 20 * len(input_seq)
#         height = 60
#         image = Image.new("RGB", (width, height), color="white")
#         draw = ImageDraw.Draw(image)

#         for j, char in enumerate(input_seq):
#             w = int(weights[j] * 255)
#             color = (255, 255 - w, 255 - w)
#             draw.rectangle([j*20, 0, (j+1)*20, height], fill=color)
#             draw.text((j*20+5, height//2 - 10), char, font=font, fill="black")

#         draw.text((5, 5), f"Output: {output_char}", fill="black", font=font)
#         frames.append(image)

#     frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=delay, loop=0)
#     temp_path = f"temp_frame_{i}.png"
#     plt.savefig(temp_path)
#     images.append(imageio.imread(temp_path))
#     os.remove(temp_path)

#     imageio.mimsave(out_path, images, duration=0.8)

def draw_connectivity1old(attention, input_seq, output_seq):
    """
    Draw static connectivity figure showing highest attention connections.
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()

    for i, out_char in enumerate(output_seq):
        for j, in_char in enumerate(input_seq):
            weight = attention[i][j].item()
            if weight > 0.1:  # threshold
                G.add_edge(f"{in_char}_{j}", f"{out_char}_{i}", weight=weight)

    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, edge_color=weights, edge_cmap=plt.cm.Greens, node_size=1500, font_size=10)
    plt.title("Character Connectivity via Attention")
    plt.show()

# geepsuggeste
def plot_attention(attn_weights, input_seq, output_seq, fname):

    fig, ax = plt.subplots(figsize=(10, 8))
    # sns.heatmap(attn_weights, xticklabels=input_seq, yticklabels=output_seq, ax=ax)
    sns.heatmap(attn_weights.detach().cpu().numpy(), xticklabels=input_seq, yticklabels=output_seq, ax=ax)

    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Attention Heatmap')
    plt.savefig(fname)

def log_attention_table(attn_list, input_list, output_list):
    devanagari_font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf")

    heat_table = wandb.Table(columns=["Input", "Output", "Attention Heatmap"])
    for i in range(min(len(attn_list), 10)):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(attn_list[i], xticklabels=input_list[i], yticklabels=output_list[i], cmap="viridis", ax=ax)
        ax.set_title(f"Sample {i}")
        plt.tight_layout()
        heat_table.add_data(''.join(input_list[i]), ''.join(output_list[i]), wandb.Image(fig))

        # Apply Devanagari font to ticks
        ax.set_xticklabels(input_list[i], fontproperties=devanagari_font, rotation=90)
        ax.set_yticklabels(output_list[i], fontproperties=devanagari_font, rotation=0)

        ax.set_title(f"Sample {i}", fontproperties=devanagari_font)
        ax.set_xlabel("Input (Latin)", fontproperties=devanagari_font)
        ax.set_ylabel("Output (Devanagari)", fontproperties=devanagari_font)
        plt.tight_layout()

        heat_table.add_data(''.join(input_list[i]), ''.join(output_list[i]), wandb.Image(fig))
        plt.close(fig)
    wandb.log({"Attention Heatmaps Table": heat_table})