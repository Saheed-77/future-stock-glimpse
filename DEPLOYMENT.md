# Deployment Guide

This guide will help you deploy your Stock Prediction app with the frontend on Vercel and the backend on Render.

## Backend Deployment (Render)

### 1. Prepare Your Backend

1. Make sure all your backend files are in the `backend/` directory
2. Ensure `requirements.txt` is up to date
3. The `Procfile` is already created for Render

### 2. Deploy to Render

1. Go to [Render.com](https://render.com) and sign up/login
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`

### 3. Set Environment Variables in Render

Go to your service's Environment tab and add:

```
FLASK_ENV=production
FRONTEND_URL=https://your-frontend-app.vercel.app
```

Replace `your-frontend-app` with your actual Vercel app name.

### 4. Get Your Backend URL

After deployment, Render will provide a URL like:
`https://your-backend-name.onrender.com`

## Frontend Deployment (Vercel)

### 1. Deploy to Vercel First

1. Go to [Vercel.com](https://vercel.com) and sign up/login
2. Click "New Project"
3. Import your project from GitHub
4. Vercel will automatically detect it's a Vite project
5. Click "Deploy" (it will use default settings initially)

### 2. Set Environment Variables in Vercel

After your initial deployment:

1. Go to your project dashboard in Vercel
2. Click on "Settings" tab
3. Click on "Environment Variables" in the sidebar
4. Add the following variable:

```
Name: VITE_API_BASE_URL
Value: https://your-backend-name.onrender.com/api
```

**Important**: Replace `your-backend-name` with your actual Render service name.

### 3. Redeploy

1. After adding the environment variable, go to the "Deployments" tab
2. Click the three dots (...) on the latest deployment
3. Click "Redeploy" to apply the new environment variable

### 4. Get Your Frontend URL

Your Vercel app will be available at a URL like:
`https://your-app.vercel.app` or `https://your-repo-name.vercel.app`

### 3. Update Backend CORS

1. Go back to your Render service
2. Update the `FRONTEND_URL` environment variable with your actual Vercel URL:
   ```
   FRONTEND_URL=https://your-app.vercel.app
   ```
3. Redeploy your backend service

## Local Development

For local development, use the provided environment files:

### Frontend (.env.local)
```
VITE_API_BASE_URL=http://localhost:5000/api
```

### Backend (.env)
```
FLASK_ENV=development
FRONTEND_URL=http://localhost:8080
```

## Testing the Deployment

1. Visit your Vercel frontend URL
2. Test the stock prediction features
3. Check browser console for any API errors
4. Verify CORS is working correctly

## Troubleshooting

### Common Issues:

1. **CORS Errors**: Make sure `FRONTEND_URL` in Render matches your Vercel URL exactly
2. **API Not Found**: Ensure `VITE_API_BASE_URL` points to your Render backend with `/api` suffix
3. **Build Failures**: Check that all dependencies are in `requirements.txt`
4. **Environment Variables**: Ensure all required environment variables are set in both platforms
5. **"Environment Variable references Secret which does not exist"**: 
   - This happens when `vercel.json` references secrets incorrectly
   - Environment variables should be set in Vercel dashboard, not in `vercel.json`
   - The `vercel.json` file should not contain `env` or `build.env` sections referencing secrets
6. **Environment variables not working**: 
   - Make sure to redeploy after adding environment variables in Vercel
   - Environment variables in Vite must start with `VITE_`
   - Check that `import.meta.env.VITE_API_BASE_URL` is accessible in your code

### Logs:
- **Render**: Check logs in your Render dashboard
- **Vercel**: Check function logs in your Vercel dashboard
- **Browser**: Check Network tab in developer tools

## Environment Variables Summary

### Vercel Environment Variables:
- `VITE_API_BASE_URL`: Your Render backend URL + `/api`

### Render Environment Variables:
- `FLASK_ENV`: `production`
- `FRONTEND_URL`: Your Vercel frontend URL
- `PORT`: Automatically set by Render
- `HOST`: Automatically set by Render

## Security Notes

- Never commit `.env` files to version control
- Use `.env.example` files as templates
- Keep your production URLs private
- Regularly update dependencies for security patches
