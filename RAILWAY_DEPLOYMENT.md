# 🚀 Railway Deployment Guide for Wyckoff Chatbot

## Quick Start Commands

```bash
# 1. Login to Railway (opens browser)
railway login

# 2. Initialize project
railway init

# 3. Deploy your application
railway up
```

## 📋 Pre-Deployment Checklist

✅ **Model files in place**: `assets/transformer_chatbot_gpu_deco_2.pth`  
✅ **Dataset file ready**: `assets/Cleaned_Wyckoff_QA_Dataset.csv`  
✅ **Dockerfile optimized** for Railway  
✅ **Environment variables configured**  
✅ **Railway.json configuration** ready  

## 🎯 Step-by-Step Deployment

### Step 1: Authentication
```bash
railway login
```
- This opens your browser for GitHub authentication
- Authorize Railway to access your repositories

### Step 2: Initialize Project
```bash
railway init
```
- Choose "Deploy from GitHub repo"
- Select your Wyckoff_chatbot repository
- Railway will create a new project

### Step 3: Configure Environment Variables
In the Railway dashboard or via CLI:
```bash
railway variables set FLASK_ENV=production
railway variables set PORT=5000
railway variables set CUDA_VISIBLE_DEVICES=0
```

### Step 4: Deploy
```bash
railway up
```
- Railway will build your Docker container
- Deploy with GPU support
- Provide you with a live URL

## 🔧 Railway Configuration

Your `railway.json` is configured for:
- **Docker build** with GPU support
- **Health checks** on the root path
- **Automatic restarts** on failure
- **Production environment** variables

## 💰 Railway Pricing

### Free Tier:
- $5/month credit
- Basic CPU instances
- Limited build time

### Pro Tier:
- $20/month
- GPU support available
- Faster builds
- More resources

## 🚨 Important Notes

### GPU Support:
- Railway supports GPU instances
- Enable in project settings
- Additional cost for GPU usage

### Model Files:
- Your `.pth` file is ~721MB
- Railway handles large files well
- Consider model optimization for faster loading

### Environment Variables:
```bash
FLASK_ENV=production
PORT=5000
CUDA_VISIBLE_DEVICES=0
```

## 🔍 Monitoring Your Deployment

### Railway Dashboard:
- View logs in real-time
- Monitor resource usage
- Check deployment status
- Manage environment variables

### Health Checks:
- Railway monitors `/` endpoint
- Automatic restarts on failure
- 100-second timeout configured

## 🛠️ Troubleshooting

### Common Issues:

1. **Build Fails**:
   ```bash
   # Check logs
   railway logs
   
   # Rebuild
   railway up --detach
   ```

2. **Model Loading Issues**:
   - Verify file paths in `app.py`
   - Check file permissions
   - Ensure files are in `assets/` folder

3. **GPU Not Available**:
   - Enable GPU in Railway dashboard
   - Check `CUDA_VISIBLE_DEVICES` variable
   - Verify CUDA installation in container

4. **Memory Issues**:
   - Monitor resource usage
   - Consider model optimization
   - Use CPU fallback if needed

### Debug Commands:
```bash
# View logs
railway logs

# Check status
railway status

# View variables
railway variables

# Connect to container
railway shell
```

## 🎉 Post-Deployment

### Your App Will Be Available At:
- Railway provides a unique URL
- Format: `https://your-app-name.railway.app`
- HTTPS enabled automatically

### Next Steps:
1. **Test the chatbot** functionality
2. **Verify GPU acceleration** is working
3. **Monitor performance** in Railway dashboard
4. **Set up custom domain** (optional)

## 📊 Performance Optimization

### For Better Performance:
1. **Enable GPU** in Railway settings
2. **Monitor memory usage**
3. **Optimize model loading**
4. **Use caching** for repeated requests

### Cost Optimization:
1. **Monitor usage** in Railway dashboard
2. **Scale down** during low traffic
3. **Use CPU fallback** for non-critical requests
4. **Optimize model size** if possible

## 🆘 Support

### Railway Support:
- Documentation: https://docs.railway.app
- Community: Railway Discord
- Status: https://status.railway.app

### Your App Logs:
```bash
railway logs --follow
```

## 🎯 Success Indicators

✅ **Deployment successful** when you see:
- "Deployment complete" message
- Live URL provided
- Health check passing
- No error logs

✅ **GPU working** when you see:
- CUDA device detected in logs
- Model loads successfully
- Fast inference times

Your Wyckoff chatbot is now live on Railway! 🚀
