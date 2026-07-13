"""
Premium styling theme and custom CSS injection for DeepAgent Harness.
Implements dark mode aesthetics, glassmorphism panels, customized scrollbars,
and modern font faces.
"""
import streamlit as st

def inject_premium_styles():
    """Inject high-end custom CSS into the Streamlit app."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap" rel="stylesheet">
        
        <style>
        /* Global Reset and Fonts */
        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #f1f5f9;
        }
        
        code, pre, [data-testid="stCode"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 14px !important;
        }

        /* Dark Slate & Glassmorphism Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            border: 1px solid rgba(99, 102, 241, 0.3);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.05);
            transform: translateY(-2px);
        }
        
        /* Agent Card Style */
        .agent-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        /* Custom Header Styling */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }
        
        /* Dynamic Accent Titles */
        .gradient-text {
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        /* Glowing effect */
        .glow-border {
            border: 1px solid #6366f1;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
        }
        
        /* Timeline style */
        .timeline-item {
            padding-left: 20px;
            border-left: 2px solid #334155;
            position: relative;
            margin-bottom: 12px;
        }
        
        .timeline-item::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #6366f1;
            border-radius: 50%;
            position: absolute;
            left: -5px;
            top: 6px;
        }
        
        /* Custom buttons styling */
        .stButton>button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        /* Chat bubble formatting overrides */
        .stChatMessage {
            background-color: rgba(30, 41, 59, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.03) !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
        }
        
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            background-color: rgba(99, 102, 241, 0.05) !important;
            border: 1px solid rgba(99, 102, 241, 0.1) !important;
        }

        /* Tool Outputs Expander Header styling */
        .st-emotion-cache-p4m0vl {
            background-color: rgba(15, 23, 42, 0.6) !important;
        }
        
        /* Hide default Streamlit footer/header details if desired */
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
