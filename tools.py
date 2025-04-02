import json
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image as PILimage
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, MODEL_NAME

# Initialize Anthropic client (used by both tools.py and simple_truss_agent.py)
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Define the tools available to the agent
TOOLS_DESCRIPTION = [
    {
        "name": "analyze_image",
        "description": "Extract truss structure data from an image using Claude Vision. Returns a JSON object with nodes (coordinates), elements (connections), forces, constraints, and material properties (E and A).",
        "input_schema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "object",
                    "description": "PIL Image object containing the truss structure"
                }
            },
            "required": []
        }
    },
    {
        "name": "plot_truss",
        "description": "Visualize truss structure with numbered nodes, elements, forces, and constraints.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "description": "List of [x, y] node coordinates"
                },
                "elements": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "description": "List of [node1_idx, node2_idx] element connections"
                },
                "forces": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "description": "Optional list of [node_idx, fx, fy] force vectors"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                    "description": "Optional list of [node_idx, dx, dy] boundary conditions"
                }
            },
            "required": ["nodes", "elements"]
        }
    },
    {
        "name": "run_analysis",
        "description": "Perform FEA to calculate displacements and stresses in the truss structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "description": "List of [x, y] node coordinates"
                },
                "elements": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "description": "List of [node1_idx, node2_idx] element connections"
                },
                "forces": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "description": "List of [node_idx, fx, fy] force vectors"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                    "description": "List of [node_idx, dx, dy] boundary conditions (0=fixed, 1=free)"
                },
                "A": {
                    "type": "number",
                    "description": "Cross-sectional area (m²)"
                },
                "E": {
                    "type": "number",
                    "description": "Young's modulus (Pa)"
                }
            },
            "required": ["nodes", "elements", "forces", "constraints", "A", "E"]
        }
    }
]

def analyze_image(image=None):
    """Use Claude Vision to extract truss data from the uploaded image"""
    if image is None:
        return None

    try:
        # Convert PIL image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Call Claude Vision
        prompt_image = """
        Analyze this image of a truss structure and extract the following information:

        1. Node locations (coordinates)
        2. Elements (connections between nodes)
        3. Forces applied to nodes
        4. Constraints (boundary conditions)
        5. Material properties (E and A) if found.

        Return in JSON format ONLY with the following format:

        {
          "nodes": [[x1, y1], [x2, y2], ...],
          "elements": [[node1, node2], [node3, node4], ...],
          "forces": [[node, fx, fy], ...],
          "constraints": [[node, cx, cy], ...],
          "E": null,
          "A": null
        }

        Where:
        - Nodes are indexed starting from 0
        - For constraints: cx/cy = 0 means fixed, cx/cy = 1 means free
        - Use metric units: meters (m) for distances and newtons (N) for forces
        - x axis is horizontal, y axis is vertical
        - For forces with angles, calculate the x and y components and provide the actual numbers
        """

        SYSTEM_PROMPT = """You are a JSON-only response bot. Your responses must:
        1. Strictly follow the format of the JSON structure.
        2. Provide numbers in meters for distances and newtons for forces.
        2. Never add any text or comments. 
        3. Never use markdown formatting or code blocks, or symbols like \n
        4. Always provide numerical values for forces, never mathematical expressions.
        5. For forces with angles, calculate and provide the actual x and y components.
    
        """

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2000,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_str
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_image
                        }
                    ]
                }
            ]
        )

        # Extract and parse the JSON from the response
        response_text = response.content[0].text
        
        # Find the JSON content between curly braces
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_content = response_text[json_start:json_end]
        else:
            raise ValueError("No valid JSON found in response")

        truss_data = json.loads(json_content)

        # Ensure E and A exist at root level
        if "E" not in truss_data:
            truss_data["E"] = None
        if "A" not in truss_data:
            truss_data["A"] = None

        # Remove material_properties if it exists
        truss_data.pop("material_properties", None)

        return truss_data

    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def plot_truss(nodes, elements, forces=None, constraints=None, **kwargs):
    """Plot a 2D truss structure with numbered nodes and elements
    
    Args:
        nodes (list): List of [x,y] node coordinates
        elements (list): List of [node1,node2] element connections
        forces (list, optional): List of [node,fx,fy] force vectors
        constraints (list, optional): List of [node,dx,dy] boundary conditions
    
    Returns:
        PIL.Image: Image object containing the plot
    """
    fig, ax = _setup_truss_plot()
    
    _plot_nodes_and_elements(ax, nodes, elements, style='initial')
    
    if constraints is not None:
        _plot_boundary_conditions(ax, nodes, constraints)
        
    if forces is not None:
        _plot_forces(ax, nodes, forces)
        
    ax.set_title('Truss Structure')
    ax.autoscale()
    ax.margins(0.1)
    
    # Set aspect ratio after autoscaling
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return PILimage.open(buf)

def calculate_max_stresses(stresses):
    """Calculate maximum tension, compression, and absolute stresses in MPa"""
    max_tension = max(max(0, stress) for stress in stresses)
    max_compression = min(min(0, stress) for stress in stresses)
    abs_max_stress = max(abs(max_tension), abs(max_compression))
    return max_tension, max_compression, abs_max_stress

def run_analysis(nodes, elements, forces, constraints, A, E):
    """Run FEM analysis on the truss structure
    
    Args:
        nodes (list): List of [x,y] node coordinates
        elements (list): List of [node1,node2] element connections
        forces (list): List of [node,fx,fy] force vectors
        constraints (list): List of [node,dx,dy] boundary conditions
        A (float): Cross-sectional area in m²
        E (float): Young's modulus in Pa
        
    Returns:
        tuple: (max_displacement, max_tension, max_compression, abs_max_stress, result_image)
    """
    try:
        num_nodes = len(nodes)
        ndof = 2 * num_nodes
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        # Assemble global stiffness matrix
        for n1, n2 in elements:
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            L = np.hypot(x2 - x1, y2 - y1)
            c, s = (x2 - x1) / L, (y2 - y1) / L

            k = (A * E / L) * np.array([
                [c*c,  c*s, -c*c, -c*s],
                [c*s,  s*s, -c*s, -s*s],
                [-c*c, -c*s, c*c,  c*s],
                [-c*s, -s*s, c*s,  s*s]
            ])
            dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += k[i, j]

        # Apply forces
        for node, fx, fy in forces:
            F[2*node] += fx
            F[2*node+1] += fy

        # Apply DOF constraints
        constrained_dofs = []
        for node, dx, dy in constraints:
            if dx == 0:
                constrained_dofs.append(2*node)
            if dy == 0:
                constrained_dofs.append(2*node+1)

        free_dofs = list(set(range(ndof)) - set(constrained_dofs))

        # Solve the system
        K_red = K[np.ix_(free_dofs, free_dofs)]
        F_red = F[free_dofs]
        
        try:
            U_red = np.linalg.solve(K_red, F_red)
        except np.linalg.LinAlgError:
            print("Error: Singular matrix. The structure might be unstable.")
            return None, None, None, None, None
        except Exception as e:
            print(f"Error solving system: {str(e)}")
            return None, None, None, None, None

        # Reconstruct full displacement vector
        U = np.zeros(ndof)
        U[free_dofs] = U_red

        # Calculate element forces and stresses
        stresses = []
        for n1, n2 in elements:
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            L = np.hypot(x2 - x1, y2 - y1)
            c, s = (x2 - x1) / L, (y2 - y1) / L
            u = U[[2*n1, 2*n1+1, 2*n2, 2*n2+1]]
            force = (A * E / L) * np.array([-c, -s, c, s]) @ u
            stress = force / A
            stresses.append(stress / 1e6)  # Convert Pa to MPa

        # Reshape U into the desired format
        U_reshaped = [[U[2*i], U[2*i+1]] for i in range(len(nodes))]

        # Calculate maximum values
        max_displacement = np.max(np.abs(U))
        max_tension, max_compression, abs_max_stress = calculate_max_stresses(stresses)

        # Create result image by directly passing all data
        result_image = plot_results(nodes, elements, U_reshaped, stresses, constraints, forces)

        # Return structured data
        analysis_data = {
            "status": "success",
            "analysis_results": {
                "displacement": float(max_displacement),
                "tension": float(max_tension),
                "compression": float(max_compression),
                "max_stress": float(abs_max_stress),
                "material": {
                    "E": float(E),
                    "A": float(A)
                }
            }
        }

        return analysis_data, result_image
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return {"status": "error", "message": str(e)}, None

def plot_results(nodes, elements, U, stresses, constraints, forces):
    """Plot analysis results with deformed shape and stresses
    
    Args:
        nodes (list): List of [x,y] node coordinates
        elements (list): List of [node1,node2] element connections
        U (list): List of [dx,dy] displacements for each node
        stresses (list): List of stresses for each element
        constraints (list): List of [node,dx,dy] boundary conditions
        forces (list): List of [node,fx,fy] force vectors
        
    Returns:
        PIL.Image: Image object containing the plot
    """
    # Check if we have all required data
    if any(x is None for x in [nodes, elements, U, stresses, constraints, forces]):
        print("Error: Missing required data for plotting results")
        return None

    fig, ax = _setup_truss_plot(figsize=(12, 8))
    
    scale = 0.1 * np.max(np.abs(nodes)) / np.max(np.abs(U))
    deformed_nodes = np.array(nodes) + scale * np.array(U).reshape(-1, 2)
    
    # Plot original and deformed structure
    orig_lines = [np.array(nodes)[[n1, n2]] for n1, n2 in elements]
    deformed_lines = [deformed_nodes[[n1, n2]] for n1, n2 in elements]
    
    # Original structure (dashed)
    lc_orig = LineCollection(orig_lines, linestyles='dashed', 
                           colors='gray', linewidths=1)
    ax.add_collection(lc_orig)
    
    # Deformed structure with stresses
    lc_deformed = LineCollection(deformed_lines, cmap='coolwarm', linewidths=5)
    lc_deformed.set_array(np.array(stresses))
    ax.add_collection(lc_deformed)
    
    # Add colorbar
    fig.colorbar(lc_deformed, label='Stress (MPa)')
    
    # Plot nodes
    ax.scatter(*zip(*nodes), c='k', s=30, zorder=3)
    ax.scatter(*zip(*deformed_nodes), c='r', s=20, zorder=3)
    
    _plot_boundary_conditions(ax, nodes, constraints, deformed_nodes)
    _plot_forces(ax, nodes, forces, deformed_nodes=deformed_nodes)
    
    ax.set_title('Truss Structure - FEM Analysis')
    ax.autoscale()
    ax.margins(0.1)
    
    # Set aspect ratio after autoscaling
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return PILimage.open(buf)

def _setup_truss_plot(figsize=(10, 8)):
    """Create and setup the basic matplotlib figure and axes for truss plotting
    
    Args:
        figsize (tuple): Figure size in inches
        
    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlabel='X', ylabel='Y')
    ax.grid(True)
    return fig, ax

def _plot_nodes_and_elements(ax, nodes, elements, style='initial'):
    """Plot nodes and elements with optional numbering
    
    Args:
        ax: Matplotlib axes
        nodes: List of [x,y] coordinates
        elements: List of [n1,n2] connections
        style: Either 'initial' for first plot or 'analysis' for results
    """
    # Plot nodes
    ax.scatter(*zip(*nodes), 
              color='blue' if style == 'initial' else 'k',
              s=50, 
              zorder=3)
    
    if style == 'initial':
        # Add node numbers
        for i, (x, y) in enumerate(nodes):
            ax.text(x, y, f' {i}', fontsize=12, va='bottom', color='blue')
            
        # Plot elements with numbers
        for i, (n1, n2) in enumerate(elements):
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, f' {i}', fontsize=10, va='bottom', 
                   ha='center', color='black',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

def _plot_boundary_conditions(ax, nodes, constraints, deformed_nodes=None):
    """Plot boundary condition symbols
    
    Args:
        ax: Matplotlib axes
        nodes: List of [x,y] coordinates
        constraints: List of [node,dx,dy] boundary conditions
        deformed_nodes: Optional deformed node positions for analysis plot
    """
    plot_nodes = deformed_nodes if deformed_nodes is not None else nodes
    for node, dx, dy in constraints:
        x, y = plot_nodes[node]
        marker = '^' if dx == dy == 0 else 'o'
        ax.plot(x, y, marker, markersize=15, zorder=4, color='k')

def _plot_forces(ax, nodes, forces, scale_factor=None, deformed_nodes=None):
    """Plot force arrows and labels
    
    Args:
        ax: Matplotlib axes
        nodes: List of [x,y] coordinates
        forces: List of [node,fx,fy] forces
        scale_factor: Optional scale factor for force arrows
        deformed_nodes: Optional deformed node positions for analysis plot
    """
    plot_nodes = deformed_nodes if deformed_nodes is not None else nodes
    
    # Calculate force scale if not provided
    if scale_factor is None:
        max_coord = np.max(np.abs(nodes))
        force_vectors = np.array([[fx, fy] for _, fx, fy in forces])
        max_force = np.linalg.norm(force_vectors, axis=1).max()
        scale_factor = 0.2 * max_coord / max_force
    
    for node, fx, fy in forces:
        x, y = plot_nodes[node]
        dx, dy = scale_factor * fx, scale_factor * fy
        ax.arrow(x, y, dx, dy, color='red', width=0.03, 
                head_width=0.15, head_length=0.2, zorder=2)
        force_mag = np.sqrt(fx**2 + fy**2)
        ax.text(x + dx/2, y + dy/2, f'{force_mag:.0f} N', 
                color='black', fontweight='bold', zorder=2,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))