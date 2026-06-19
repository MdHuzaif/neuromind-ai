"""
NeuroMind AI - Neuron Simulation Module
Generates Python code for LIF and Hodgkin-Huxley neuron models
"""

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# LIF Model Template
LIF_MODEL_CODE = """
import numpy as np
import matplotlib.pyplot as plt

# Leaky Integrate-and-Fire (LIF) Neuron Model
def lif_neuron_simulation(duration=1000, dt=0.1, V_rest=-65, V_thresh=-50, V_reset=-70, 
                           tau=10, R=10, I_ext=15):
    # LIF Neuron Simulation
    # Parameters:
    # - duration: Simulation time (ms)
    # - dt: Time step (ms)
    # - V_rest: Resting potential (mV)
    # - V_thresh: Threshold potential (mV)
    # - V_reset: Reset potential (mV)
    # - tau: Membrane time constant (ms)
    # - R: Membrane resistance (MΩ)
    # - I_ext: External current (nA)
    
    time = np.arange(0, duration, dt)
    V = np.zeros(len(time))
    spikes = np.zeros(len(time))
    
    V[0] = V_rest
    
    for i in range(1, len(time)):
        # LIF equation: dV/dt = -(V - V_rest)/tau + R*I_ext
        dV = (-(V[i-1] - V_rest) / tau + R * I_ext) * dt
        V[i] = V[i-1] + dV
        
        # Spike detection
        if V[i] >= V_thresh:
            spikes[i] = 1
            V[i] = V_reset  # Reset after spike
    
    return time, V, spikes

# Run simulation
time, V, spikes = lif_neuron_simulation()

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Membrane potential
ax1.plot(time, V, 'b-', linewidth=1.5)
ax1.axhline(y=-50, color='r', linestyle='--', label='Threshold')
ax1.axhline(y=-65, color='g', linestyle='--', label='Resting Potential')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Membrane Potential (mV)')
ax1.set_title('LIF Neuron - Membrane Potential')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Spike train
ax2.eventplot([time[spikes == 1]], lineoffsets=0, linelengths=1, colors='red')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Spikes')
ax2.set_title('LIF Neuron - Spike Train')
ax2.set_xlim(0, max(time))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Total spikes: {int(np.sum(spikes))}")
print(f"Firing rate: {np.sum(spikes) / (max(time)/1000):.2f} Hz")
"""

# Hodgkin-Huxley Model Template
HH_MODEL_CODE = """
import numpy as np
import matplotlib.pyplot as plt

# Hodgkin-Huxley Neuron Model
def hodgkin_huxley_simulation(duration=100, dt=0.01, I_ext=10):
    # Hodgkin-Huxley Neuron Model
    # Parameters:
    # - duration: Simulation time (ms)
    # - dt: Time step (ms)
    # - I_ext: External current (µA/cm²)
    
    # Membrane parameters
    C_m = 1.0  # Membrane capacitance (µF/cm²)
    
    # Conductances (mS/cm²)
    g_Na = 120.0  # Sodium
    g_K = 36.0    # Potassium
    g_L = 0.3     # Leak
    
    # Reversal potentials (mV)
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387
    
    # Initialize
    time = np.arange(0, duration, dt)
    V = np.zeros(len(time))
    m = np.zeros(len(time))  # Na activation
    h = np.zeros(len(time))  # Na inactivation
    n = np.zeros(len(time))  # K activation
    
    # Initial conditions
    V[0] = -65.0
    m[0] = 0.05
    h[0] = 0.6
    n[0] = 0.32
    
    # Rate functions
    def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
    def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
    def beta_h(V): return 1.0 / (1 + np.exp(-(V + 35) / 10))
    def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)
    
    # Simulation
    for i in range(1, len(time)):
        # Ionic currents
        I_Na = g_Na * (m[i-1]**3) * h[i-1] * (V[i-1] - E_Na)
        I_K = g_K * (n[i-1]**4) * (V[i-1] - E_K)
        I_L = g_L * (V[i-1] - E_L)
        
        # Membrane potential update
        dV = (I_ext - I_Na - I_K - I_L) / C_m * dt
        V[i] = V[i-1] + dV
        
        # Gating variables update
        dm = (alpha_m(V[i-1]) * (1 - m[i-1]) - beta_m(V[i-1]) * m[i-1]) * dt
        dh = (alpha_h(V[i-1]) * (1 - h[i-1]) - beta_h(V[i-1]) * h[i-1]) * dt
        dn = (alpha_n(V[i-1]) * (1 - n[i-1]) - beta_n(V[i-1]) * n[i-1]) * dt
        
        m[i] = m[i-1] + dm
        h[i] = h[i-1] + dh
        n[i] = n[i-1] + dn
    
    return time, V, m, h, n

# Run simulation
time, V, m, h, n = hodgkin_huxley_simulation()

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Membrane potential
axes[0].plot(time, V, 'b-', linewidth=1.5)
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Membrane Potential (mV)')
axes[0].set_title('Hodgkin-Huxley Neuron - Action Potential')
axes[0].grid(True, alpha=0.3)

# Gating variables
axes[1].plot(time, m, 'r-', label='m (Na activation)', linewidth=1.5)
axes[1].plot(time, h, 'g-', label='h (Na inactivation)', linewidth=1.5)
axes[1].plot(time, n, 'b-', label='n (K activation)', linewidth=1.5)
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Gating Variable')
axes[1].set_title('Hodgkin-Huxley Neuron - Gating Variables')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Peak membrane potential: {np.max(V):.2f} mV")
print(f"Minimum membrane potential: {np.min(V):.2f} mV")
"""

def generate_neuron_code(user_query: str, context: str, groq_api_key: str) -> dict:
    """
    Generate Python code for neuron simulation based on user query
    """
    prompt_template = """
    You are an expert Computational Neuroscientist specializing in neuron modeling.
    
    Context from research papers:
    {context}
    
    User Request: {query}
    
    Generate a complete, executable Python script for neuron simulation. 
    STRICT RULES:
    1. Use ONLY these libraries: numpy, matplotlib.pyplot
    2. The code MUST be enclosed in a single ```python ... ``` block.
    3. Implement either Leaky Integrate-and-Fire (LIF) or Hodgkin-Huxley model based on user request.
    4. Include realistic parameters (membrane potential, threshold, time constants).
    5. Add clear comments explaining the neuroscience concepts.
    6. Create publication-quality plots with proper labels and titles.
    7. Print key metrics (firing rate, spike count, peak potential).
    8. Do NOT include any markdown text outside the code block.
    
    If user asks for dopamine modulation, synaptic plasticity, or specific brain regions,
    modify the parameters accordingly (e.g., change threshold, time constant, or external current).
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048
        )
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "query": user_query})
        raw_code = response.content
        
        # Extract code block
        match = re.search(r"```python\n(.*?)\n```", raw_code, re.DOTALL)
        if match:
            clean_code = match.group(1).strip()
        else:
            clean_code = raw_code.strip()
            
        return {"success": True, "code": clean_code, "error": None}
        
    except Exception as e:
        return {"success": False, "code": "", "error": str(e)}

def get_template_code(template_name: str) -> str:
    """Get pre-defined neuron simulation template"""
    templates = {
        "lif": LIF_MODEL_CODE,
        "hodgkin_huxley": HH_MODEL_CODE
    }
    return templates.get(template_name, "# Template not found")