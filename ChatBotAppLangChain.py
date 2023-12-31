# https://github.com/marshmellow77/streamlit-chatgpt-ui/blob/main/app.py
# https://towardsdatascience.com/build-your-own-chatgpt-like-app-with-streamlit-20d940417389

#import openai
from ChatEngineLangChainV2 import LanguageModel
import streamlit as st
from streamlit_chat import message


# Setting page title and header
#st.set_page_config(page_title="Pixart's ChatGPTBot PoC", page_icon=":robot_face:")
st.set_page_config(page_title="Pixart's ChatGPTBot PoC", page_icon="pixart.ico")
st.markdown("<h1 style='text-align: center;'>Pixart's ChatGPTBot PoC</h1>", unsafe_allow_html=True)

llm = LanguageModel()

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Cost")
# model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
model_name = "GPT-3.5"
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    llm.setModel("gpt-3.5-turbo")
else:
    llm.setModel("gpt-4")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):

    response = llm.response(messages=prompt)
    output = response["Answer"]
    total_tokens = response["Total Tokens"]
    prompt_tokens = response["Prompt Tokens"]
    completion_tokens = response["Completion Tokens"]
    total_cost = response["Total Cost USD"]
    return output, total_tokens, prompt_tokens, completion_tokens, total_cost


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens, total_cost = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_cost
        else:
            cost = total_cost

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Mario Rossi")
            message(st.session_state["generated"][i], key=str(i), avatar_style="initials", seed="Pixartprinting")
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")