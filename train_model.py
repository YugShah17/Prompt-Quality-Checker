"""
Root Training Wrapper for Intelligent Prompt Quality Assessment Framework.
Delegates to model/train_model.py.
"""
from model.train_model import train

if __name__ == "__main__":
    train()
