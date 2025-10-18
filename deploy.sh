#!/bin/bash

# Wyckoff Chatbot Deployment Script
# This script helps deploy the application to various platforms

echo "🚀 Wyckoff Chatbot Deployment Script"
echo "====================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to deploy to Railway
deploy_railway() {
    echo "📦 Deploying to Railway..."
    if command_exists railway; then
        railway login
        railway up
    else
        echo "❌ Railway CLI not found. Install it first:"
        echo "npm install -g @railway/cli"
    fi
}

# Function to deploy to Render
deploy_render() {
    echo "📦 Deploying to Render..."
    echo "1. Push your code to GitHub"
    echo "2. Connect your GitHub repo to Render"
    echo "3. Use the render.yaml configuration file"
    echo "4. Set environment variables in Render dashboard"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "📦 Deploying to Heroku..."
    if command_exists heroku; then
        heroku create wyckoff-chatbot-$(date +%s)
        heroku config:set FLASK_ENV=production
        git push heroku main
    else
        echo "❌ Heroku CLI not found. Install it first:"
        echo "https://devcenter.heroku.com/articles/heroku-cli"
    fi
}

# Function to deploy with Docker
deploy_docker() {
    echo "📦 Building Docker image..."
    docker build -t wyckoff-chatbot .
    echo "🐳 Running Docker container..."
    docker run -p 5000:5000 --gpus all wyckoff-chatbot
}

# Function to deploy to VPS
deploy_vps() {
    echo "📦 VPS Deployment Instructions:"
    echo "1. Set up a VPS with Ubuntu 20.04+"
    echo "2. Install Docker and Docker Compose"
    echo "3. Install NVIDIA Docker runtime for GPU support"
    echo "4. Clone your repository"
    echo "5. Run: docker-compose up -d"
}

# Main menu
echo "Select deployment option:"
echo "1) Railway (Recommended for GPU)"
echo "2) Render"
echo "3) Heroku"
echo "4) Docker (Local/VPS)"
echo "5) VPS with GPU"
echo "6) Show all options"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        deploy_railway
        ;;
    2)
        deploy_render
        ;;
    3)
        deploy_heroku
        ;;
    4)
        deploy_docker
        ;;
    5)
        deploy_vps
        ;;
    6)
        echo "All deployment options will be shown..."
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo "✅ Deployment process completed!"
