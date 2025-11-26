import json
from graph.pipeline import app

# The initial state with the user's query
state = {"query": "Should I buy Infosys stock today?"}

# Invoke the entire pipeline
final_state = app.invoke(state)

# --- This is the correct way to print the final output ---
# The clean, final data is stored in the 'report' key.
print("\n--- 🚀 FINAL REPORT 🚀 ---\n")
if 'report' in final_state:
    # Use json.dumps for pretty-printing the final dictionary
    print(json.dumps(final_state['report'], indent=2))
else:
    print("Error: 'report' key not found in the final state.")
    print("\n--- Full Final State ---")
    print(final_state)
