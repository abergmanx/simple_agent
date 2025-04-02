import re
import json
import os
from PIL import Image
import tools
from tools import client
from config import MODEL_NAME
from colorama import init, Fore, Style

# Initialize colorama for Windows compatibility
init()

class TrussState:
    """Manages the state and formatting of truss data"""
    def __init__(self):
        # Initialize all state at once - removes need for separate clear()
        self.reset()

    def reset(self):
        """Reset all state - renamed from clear() to better reflect its purpose"""
        self.data = None
        self.truss_image = None
        self.fem_image = None

    def handle_image_analysis(self, image_path):
        """Analyze image and update state. Returns (success, data)"""
        try:
            image = Image.open(image_path)
            result = tools.analyze_image(image)
            
            if not result:
                return False, None
            
            # Update state in one operation
            self.data = {**result, "original_image": image}
            self.truss_image = tools.plot_truss(**result)  # Note: passing result not self.data to avoid original_image
            self.truss_image.show()
            
            return True, self.data
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return False, None

    def get_analysis_data(self):
        """Get analysis-ready data (renamed for clarity)"""
        return {k: v for k, v in (self.data or {}).items() 
                if k != "original_image"}

    def update_material_properties(self, inputs):
        """Update E and A in truss data"""
        if not self.data:
            return inputs
            
        # Simplified property update using dict comprehension
        material_props = {k: inputs.pop(k) for k in ["E", "A"] 
                         if k in inputs}
        self.data.update(material_props)
        return inputs

    def has_material_properties(self):
        """Check if material properties are set"""
        return bool(self.data and self.data.get("E") and self.data.get("A"))

def extract_reply(text):
    """Extract the reply from Claude's response"""
    pattern = r'<reply>(.*?)</reply>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

def process_tool_call(tool_name, tool_input):
    """Process a tool call and return the result"""
    try:
        # Check if tool exists in TOOLS_DESCRIPTION
        if not any(tool["name"] == tool_name for tool in tools.TOOLS_DESCRIPTION):
            return f"Unknown tool '{tool_name}'."
            
        # Check if we have truss data for tools that need it
        if tool_name != "analyze_image" and not truss_state.data:
            return "No truss data available."
            
        if tool_name == "analyze_image":
            if not tool_input.get("image"):
                return "No image provided for analysis."
            result = tools.analyze_image(tool_input["image"])
            if result:
                truss_state.data = result
                return "Successfully analyzed image and extracted truss data."
            return "Failed to analyze image."
            
        if tool_name == "plot_truss":
            if tool_input:
                truss_state.data.update(tool_input)
            truss_state.truss_image = tools.plot_truss(**truss_state.data)
            truss_state.truss_image.show()
            return "Successfully plotted truss structure."
            
        if tool_name == "run_analysis":
            if tool_input:
                tool_input = truss_state.update_material_properties(tool_input)
                truss_state.data.update(tool_input)
                
            try:
                analysis_inputs = truss_state.get_analysis_data()
                
                # Check if material properties are missing
                if not truss_state.has_material_properties():
                    return "Material properties (E and A) must be specified before running analysis. Please provide Young's modulus (E) in Pa and cross-sectional area (A) in m²."
                    
                analysis_data, result_image = tools.run_analysis(**analysis_inputs)
                if not result_image:
                    return analysis_data  # Return error status and message
                    
                truss_state.fem_image = result_image
                truss_state.fem_image.show()
                
                # Display formatted results in terminal
                results = analysis_data["analysis_results"]
                print(f"\n{Fore.CYAN}Analysis Results:{Style.RESET_ALL}")
                print(f"- Young's modulus (E): {results['material']['E']/1e9:.2f} GPa")
                print(f"- Cross-sectional area (A): {results['material']['A']*1e6:.2f} mm²\n")
                print(f"- Max displacement: {results['displacement']:.5f} m")
                print(f"- Max tension: {results['tension']:.2f} MPa")
                print(f"- Max compression: {results['compression']:.2f} MPa")
                print(f"- Absolute max stress: {results['max_stress']:.2f} MPa\n")
                
                # Return structured data for Claude
                return analysis_data
                
            except Exception as e:
                return {"status": "error", "message": str(e)}
                
    except Exception as e:
        return {"status": "error", "message": str(e)}

# System message for the agent
SYSTEM_MESSAGE = """You are a truss structure analysis assistant. 

IMPORTANT RULES:
1. Only use tools if you have ALL required information
2. If missing any information, ask the user first
3. First think of the problem given by the user.
4. Once you have the final answer place the user-facing response in <reply></reply> tags
5. Keep responses extremely concise - one or two sentences when possible
6. Use natural language to describe changes and results

Data format:
nodes: [[x,y]], elements: [[n1,n2]], forces: [[node,fx,fy]], constraints: [[n,cx,cy]] (0=fixed,1=free)"""

# Initialize global truss state
truss_state = TrussState()

def truss_chat(image_path):
    """Simple chat interface for truss analysis
    
    Args:
        image_path: Path to the image file to analyze
    """
    # Clear any existing state
    truss_state.reset()
    
    print("\n\n========================== CHAT BOT ===================================\n\n")
    print(f"\n{Fore.CYAN}Welcome to the Truss Analysis Assistant!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Type{Style.RESET_ALL} {Fore.RED}'quit'{Style.RESET_ALL} {Fore.CYAN}to exit the chat.{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}==> {Style.RESET_ALL}Analyzing image: {image_path}")
    
    # Analyze the specified image
    if not os.path.exists(image_path):
        print(f"{Fore.RED}Error: Image file '{image_path}' not found.{Style.RESET_ALL}")
        return
        
    success, data = truss_state.handle_image_analysis(image_path)
    if not success:
        print(f"{Fore.RED}Failed to analyze image.{Style.RESET_ALL}")
        return
        
    print(f"\n{Fore.GREEN}Image analyzed successfully.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Nodes:{Style.RESET_ALL} {data.get('nodes')}")
    print(f"{Fore.CYAN}Elements:{Style.RESET_ALL} {data.get('elements')}")
    print(f"{Fore.CYAN}Forces:{Style.RESET_ALL} {data.get('forces')}")
    print(f"{Fore.CYAN}Constraints:{Style.RESET_ALL} {data.get('constraints')}")
    
    messages = []
    # Add initial message about the plot
    messages.append({
        "role": "assistant",
        "content": "I have analyzed the image and plotted the truss structure with all nodes, elements, forces, and constraints labeled."
    })
    
    while True:
        user_message = input(f"\n{Fore.GREEN}User:{Style.RESET_ALL} ")
        if user_message.lower() == "quit":
            # Clear state before exiting
            truss_state.reset()
            break
            
        # Add user message to history
        messages.append({"role": "user", "content": user_message})
        
        # Create dynamic system message with current truss data
        dynamic_system = SYSTEM_MESSAGE
        if truss_state.data:
            truss_data_json = truss_state.get_analysis_data()
            dynamic_system += f"\n\nCurrent Truss Data: {json.dumps(truss_data_json, indent=2)}"
        
        while True:  # Inner loop for handling tool calls
            # Call Claude API
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=2000,
                temperature=0,
                system=dynamic_system,
                messages=messages,
                tools=tools.TOOLS_DESCRIPTION
            )
            
            # Add Claude's response to history
            messages.append({"role": "assistant", "content": response.content})
            
            # If Claude wants to use a tool
            if response.stop_reason == "tool_use":
                # Get the tool call from the last content block
                tool_call_block = response.content[-1]
                tool_name = tool_call_block.name
                tool_input = tool_call_block.input
                
                print(f"\n{Fore.YELLOW}[AI Agent is using: {tool_name}]{Style.RESET_ALL}")
                
                # Process tool call
                tool_result = process_tool_call(tool_name, tool_input)
                
                # Add tool result to history
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_block.id,
                        "content": str(tool_result)
                    }]
                })
                
                # Only print the formatted tool result
                if isinstance(tool_result, str):
                    print(f"\n{Fore.CYAN}Result:{Style.RESET_ALL}\n{tool_result}")
                
                continue  # Continue inner loop to get Claude's response
            else:
                # Extract and print Claude's response
                text_content = "".join(
                    block.text for block in response.content 
                    if block.type == "text"
                )
                
                reply = extract_reply(text_content)
                if reply:  # Only add non-empty responses to history
                    print(f"\n{Fore.BLUE}AI Agent:{Style.RESET_ALL} {reply}")
                    messages.append({"role": "assistant", "content": text_content})
                break  # Break inner loop to get next user input

if __name__ == "__main__":
    # Specify the image path here - users can easily change this line
    image_path = "./images/book_truss.png"
    truss_chat(image_path) 