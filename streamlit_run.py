import streamlit as st
import json
import logging
import time
from graph.pipeline import app  # Import your main LangGraph app
from agents.final_summary import generate_final_summary
from pydantic import BaseModel
# Import the specific response types for isinstance checks
try:
    from agents.final_summary import StandardSummaryResponse, PortfolioSummaryResponse, SimulationSummaryResponse, ErrorSummaryResponse
except ImportError:
    st.error("Could not import response types from final_summary agent.")
    class BaseModel: pass # Define dummy base
    class StandardSummaryResponse(BaseModel): pass
    class PortfolioSummaryResponse(BaseModel): pass
    class SimulationSummaryResponse(BaseModel): pass
    class ErrorSummaryResponse(BaseModel): pass

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Investment Analyst Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Investment Analyst Chat")
st.markdown("""
Ask questions about major Indian stocks, compare assets, or discuss market scenarios.
**Disclaimer:** Educational purposes only. Not financial advice. Data may have delays.
""")

# --- Initialize Chat History ---
# Use st.session_state to store messages across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your investment analysis today?"}]

# --- Display Chat Messages ---
# Iterate through the stored messages and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if content is a dict (our structured response) or string
        if isinstance(message["content"], dict):
            # Display formatted AI response
            summary_obj = message["content"]["summary_obj"]
            final_report = message["content"].get("final_report") # Get the raw report too

            st.markdown(f"#### {summary_obj.headline}")
            st.markdown(summary_obj.summary)

            # Display based on the type of summary received using isinstance()
            if isinstance(summary_obj, StandardSummaryResponse):
                if summary_obj.key_points:
                    st.markdown("**Key Takeaways:**")
                    for point in summary_obj.key_points:
                        st.markdown(f"- {point}")
                st.info(f"**Next Steps:** {summary_obj.next_steps}")

            elif isinstance(summary_obj, (PortfolioSummaryResponse, SimulationSummaryResponse)):
                 if summary_obj.key_points:
                     st.markdown("**Key Considerations/Impacts:**")
                     for point in summary_obj.key_points:
                         st.markdown(f"- {point}")
                 st.info(f"**Next Steps:** {summary_obj.next_steps}")

            elif isinstance(summary_obj, PlaceholderResponse):
                 if hasattr(summary_obj, 'details') and summary_obj.details:
                      st.markdown("---")
                      st.markdown(summary_obj.details)

            # Always show disclaimer
            st.warning(f"**Disclaimer:** {summary_obj.disclaimer}", icon="⚠️")

            # Expander for the raw JSON report
            if final_report:
                 with st.expander("🔬 View Detailed Agent Output (JSON)"):
                      st.json(final_report)

        else:
            # Display simple text messages (initial greeting, user messages, errors)
            st.markdown(message["content"])

# --- Handle User Input ---
# Use st.chat_input for the text box at the bottom
if prompt := st.chat_input("Ask about INFY, compare stocks, or ask 'what if'..."):
    # 1. Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process the query and display AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for intermediate messages/spinner
        full_response = ""
        error_message = None
        final_report_data = None
        summary_object = None

        try:
            # --- Show spinners/progress ---
            message_placeholder.markdown("🤔 Thinking... Invoking AI agents...")
            # Ideally, use streaming here if LangGraph supports it well for your setup
            # For now, simulate steps
            time.sleep(0.5) # Simulate Router
            message_placeholder.markdown("📊 Gathering data and analyzing...")
            time.sleep(1.0) # Simulate Data/Analyst/Thesis/Verification

            # --- Run the multi-agent pipeline ---
            log.info(f"Invoking pipeline with query: {prompt}")
            initial_state = {"query": prompt}
            final_result = app.invoke(initial_state) # Use invoke

            if final_result and ('report' in final_result):
                final_report_data = final_result['report']
                log.info("Pipeline execution successful. Generating final summary.")
                message_placeholder.markdown("📝 Synthesizing final analysis...")

                # --- Generate the final conversational summary ---
                summary_object = generate_final_summary(final_report_data)

                # Check if summary generation itself resulted in an error object
                if isinstance(summary_object, ErrorSummaryResponse):
                    log.error(f"Final summary generation returned an error: {summary_object.summary}")
                    error_message = summary_object.summary
                    summary_object = None # Don't display partial error summary object directly
                else:
                    log.info("Final summary generated successfully.")

            else:
                error_message = "The analysis pipeline did not produce a final report. The query might be too complex or outside the supported scope."
                log.error(f"Pipeline execution failed or did not produce a report. Final state: {final_result}")

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            log.error(f"Error during Streamlit chat execution: {e}", exc_info=True)

        # --- Display the final AI response or error ---
        if error_message:
            message_placeholder.error(error_message)
            # Add error message to chat history
            st.session_state.messages.append({"role": "assistant", "content": error_message})
        elif summary_object:
             # Clear placeholder before displaying final content
             message_placeholder.empty()

             # Display formatted AI response (code duplicated from history display logic)
             st.markdown(f"#### {summary_object.headline}")
             st.markdown(summary_object.summary)

             if isinstance(summary_object, StandardSummaryResponse):
                 if summary_object.key_points:
                     st.markdown("**Key Takeaways:**")
                     for point in summary_object.key_points:
                         st.markdown(f"- {point}")
                 st.info(f"**Next Steps:** {summary_object.next_steps}")

             elif isinstance(summary_object, (PortfolioSummaryResponse, SimulationSummaryResponse)):
                  if summary_object.key_points:
                      st.markdown("**Key Considerations/Impacts:**")
                      for point in summary_object.key_points:
                          st.markdown(f"- {point}")
                  st.info(f"**Next Steps:** {summary_object.next_steps}")

             elif isinstance(summary_object, PlaceholderResponse):
                  if hasattr(summary_object, 'details') and summary_object.details:
                       st.markdown("---")
                       st.markdown(summary_object.details)

             st.warning(f"**Disclaimer:** {summary_object.disclaimer}", icon="⚠️")

             if final_report_data:
                  with st.expander("🔬 View Detailed Agent Output (JSON)"):
                       st.json(final_report_data)

             # Add the structured response dict to history
             st.session_state.messages.append({"role": "assistant", "content": {"summary_obj": summary_object, "final_report": final_report_data}})
        else:
             # Fallback if no error and no summary (shouldn't happen with current logic, but good practice)
             fallback_msg = "Sorry, I wasn't able to generate a response for that query."
             message_placeholder.markdown(fallback_msg)
             st.session_state.messages.append({"role": "assistant", "content": fallback_msg})