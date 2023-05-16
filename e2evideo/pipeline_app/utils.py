"""
This module contains utility functions for the streamlit app.
"""
import os
import streamlit as st


def colored_text(text, color):
    """
    This function displays colored text in the streamlit app.
    Parameters
    ----------
    text : str
        The text to be displayed.
    color : str
        The color of the text.
    Returns
    -------
    None
    """
    st.markdown(
        f"""
        <div style="color: {color}; display: inline-block;
                padding: 5px;font-family: Times New Roman;">
           <em> {text} </em>
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_subdirectories(path):
    """
    This function returns a list of subdirectories in a given path.
    Parameters
    ----------
    path : str
        The path to the directory.
    Returns
    -------
    subdirs : list
        A list of subdirectories in the given path.
    """
    subdirs = [dir_name for dir_name in os.listdir(path)
               if os.path.isdir(os.path.join(path, dir_name))]
    return subdirs

def get_parent_directory(path):
    """
    This function returns the parent directory of a given path.
    Parameters
    ----------
    path : str
        The path to the directory.
    Returns
    -------
    parent_dir : list
        A list containing the parent directory of the given path.
    """
    parent_dir = [os.path.dirname(path)]
    return parent_dir
