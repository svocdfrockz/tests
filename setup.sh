#!/bin/bash

echo "=================================================="
echo "Azure GPT-4 Deterministic Response Setup"
echo "=================================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found"

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo
    echo "⚠️  IMPORTANT: Edit .env file and add your Azure OpenAI credentials:"
    echo "   - AZURE_OPENAI_API_KEY"
    echo "   - AZURE_OPENAI_ENDPOINT"
    echo "   - AZURE_OPENAI_DEPLOYMENT"
else
    echo
    echo "✓ .env file already exists"
fi

echo
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo
echo "Next steps:"
echo "1. Edit .env with your Azure OpenAI credentials"
echo "2. Run: python simple_example.py"
echo "3. Run it again - you'll get the SAME response!"
echo "4. Run: python deterministic_gpt4.py (full demo)"
echo "5. Run: python test_deterministic.py (test suite)"
echo
