# utils/decorators.py

from typing import Any, Callable, TypeVar
import streamlit as st
import time

F = TypeVar('F', bound=Callable[..., Any])


def log_execution(func: F) -> F:
    """
    Decorator, der den Funktionsaufruf protokolliert.
    """

    def wrapper(*args, **kwargs):
        st.write(f"Calling function: {func.__name__} with args: {args[1:]}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        st.write(f"Function {func.__name__} returned: {result}")
        return result

    return wrapper  # type: ignore


def time_execution(func: F) -> F:
    """
    Decorator, der die Ausführungszeit der Funktion misst.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.write(f"Function {func.__name__} took {end_time - start_time:.4f} seconds.")
        return result

    return wrapper  # type: ignore
