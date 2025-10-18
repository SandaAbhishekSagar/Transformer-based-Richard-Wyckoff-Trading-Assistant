#!/bin/bash

echo "🚀 Deploying Wyckoff Chatbot to Railway"
echo "======================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login to Railway (interactive)
echo "🔐 Logging into Railway..."
railway login

# Initialize project if not already done
echo "📦 Initializing Railway project..."
railway init

# Set environment variables
echo "⚙️ Setting environment variables..."
railway variables set FLASK_ENV=production
railway variables set PORT=5000
railway variables set CUDA_VISIBLE_DEVICES=0

# Deploy the application
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your app will be available at the Railway URL shown above"
echo "📊 Monitor your deployment in the Railway dashboard"
