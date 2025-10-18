# Wyckoff Chatbot Deployment Guide

## 🚀 Hosting Options for Your Enhanced GPU Chatbot

Based on your memory preference for low-cost GPU hosting [[memory:5674752]], here are the best deployment options:

## Option 1: Railway (Recommended for GPU) 💰💰💰

**Cost**: $5-20/month with GPU support
**Best for**: GPU-accelerated applications

### Steps:
1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**:
   ```bash
   railway login
   railway up
   ```

3. **Configure Environment**:
   - Set `FLASK_ENV=production`
   - Enable GPU in Railway dashboard
   - Upload your model files to the `assets/` folder

4. **Benefits**:
   - Built-in GPU support
   - Automatic scaling
   - Easy deployment
   - Good for PyTorch models

---

## Option 2: Render (Budget-Friendly) 💰💰

**Cost**: $7-25/month
**Best for**: Cost-effective deployment

### Steps:
1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy to Render"
   git push origin main
   ```

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Select "Web Service"
   - Use the `render.yaml` configuration

3. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Environment: Python 3.9

---

## Option 3: VPS with GPU (Most Control) 💰💰💰

**Cost**: $20-50/month
**Best for**: Full control and GPU optimization

### Recommended Providers:
- **Vast.ai**: $0.1-0.5/hour (spot instances)
- **RunPod**: $0.2-0.8/hour
- **Lambda Labs**: $0.5-1.5/hour
- **Paperspace**: $0.45-0.9/hour

### Steps:
1. **Choose a VPS Provider**:
   - Sign up for a GPU-enabled instance
   - Select Ubuntu 20.04+ with CUDA support

2. **Setup Server**:
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install NVIDIA Docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Deploy Application**:
   ```bash
   # Clone repository
   git clone <your-repo-url>
   cd Wyckoff_chatbot
   
   # Build and run with Docker Compose
   docker-compose up -d
   ```

4. **Configure Nginx** (Optional):
   ```bash
   sudo apt install nginx
   # Configure reverse proxy to port 5000
   ```

---

## Option 4: Heroku (Limited GPU) 💰💰

**Cost**: $7-25/month
**Best for**: Quick deployment without GPU

### Steps:
1. **Install Heroku CLI**:
   ```bash
   # Windows
   winget install Heroku.HerokuCLI
   
   # Or download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy**:
   ```bash
   heroku create wyckoff-chatbot-$(date +%s)
   heroku config:set FLASK_ENV=production
   git push heroku main
   ```

3. **Note**: Heroku doesn't support GPU, so your model will run on CPU

---

## Option 5: Google Cloud Run (Serverless) 💰💰💰

**Cost**: Pay-per-use (very cheap for low traffic)
**Best for**: Serverless deployment

### Steps:
1. **Setup Google Cloud**:
   ```bash
   # Install gcloud CLI
   # Create a new project
   gcloud projects create wyckoff-chatbot
   gcloud config set project wyckoff-chatbot
   ```

2. **Deploy with Cloud Run**:
   ```bash
   # Build and push to Google Container Registry
   gcloud builds submit --tag gcr.io/wyckoff-chatbot/app
   
   # Deploy to Cloud Run
   gcloud run deploy --image gcr.io/wyckoff-chatbot/app --platform managed
   ```

---

## 🎯 Recommended Deployment Strategy

### For Development/Testing:
- **Render** or **Heroku** (CPU-only, cheap)

### For Production with GPU:
- **Railway** (easiest GPU setup)
- **VPS with GPU** (most control, best performance)

### For Maximum Cost Savings:
- **Vast.ai** or **RunPod** (spot instances)
- **Google Cloud Run** (pay-per-use)

---

## 🔧 Pre-Deployment Checklist

1. **✅ Fixed hardcoded paths** in `app.py`
2. **✅ Created deployment configs** for all platforms
3. **✅ Added production environment handling**
4. **📋 Ensure model files are in `assets/` folder**
5. **📋 Test locally with `python app.py`**

---

## 🚨 Important Notes

### GPU Requirements:
- Your app uses PyTorch with CUDA
- Most cloud platforms charge extra for GPU instances
- Consider CPU fallback for cost optimization

### Model Files:
- Ensure `transformer_chatbot_gpu_deco_2.pth` is in `assets/`
- Ensure `Cleaned_Wyckoff_QA_Dataset.csv` is in `assets/`
- These files are large and may need special handling

### Environment Variables:
```bash
FLASK_ENV=production
PORT=5000
CUDA_VISIBLE_DEVICES=0  # For GPU deployment
```

---

## 💡 Cost Optimization Tips

1. **Use spot instances** on Vast.ai/RunPod (up to 90% cheaper)
2. **Implement model caching** to reduce inference time
3. **Use CPU fallback** for non-critical requests
4. **Monitor usage** and scale down during low traffic

---

## 🆘 Troubleshooting

### Common Issues:
1. **Model loading fails**: Check file paths and permissions
2. **GPU not detected**: Verify CUDA installation
3. **Memory issues**: Reduce batch size or use CPU
4. **Timeout errors**: Increase timeout settings

### Support:
- Check logs: `docker logs <container-name>`
- Monitor resources: `nvidia-smi` (for GPU)
- Test locally first: `python app.py`

---

## 🎉 Quick Start Commands

```bash
# Test locally
python app.py

# Deploy with Docker
docker-compose up -d

# Deploy to Railway
railway up

# Deploy to Render
# Just push to GitHub and connect to Render
```

Choose the option that best fits your budget and requirements! 🚀
