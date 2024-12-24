# # Stage 1: Build
# FROM node:18-alpine AS build
#
# # declare build time variable for Docker
# ARG REACT_APP_API_URL
#
# # declare environment variable for React
# ENV REACT_APP_API_URL=$REACT_APP_API_URL
#
# # Stage 1: Build
# # FROM node:18-alpine AS build
#
# # (Build stage remains the same...)
# WORKDIR /app
# COPY package.json package-lock.json ./
# RUN npm ci
# COPY . .
# RUN npm run build
#
# # Stage 2: Serve with Node (HTTP ONLY)
# FROM node:18-alpine AS prod
# WORKDIR /app
# # 1. Install the 'serve' package instead of 'http-server'
# RUN npm install -g serve
# COPY --from=build /app/build ./build
#
# EXPOSE 3000
# # 2. Use 'serve' with the '-s' flag to handle SPA routing
# CMD ["serve", "-s", "build", "-l", "3000"]
#
# WORKDIR /app
# COPY package.json package-lock.json ./
# RUN npm ci
# COPY . .
# RUN npm run build
#
#
# # Stage 2: Serve with Node 
# FROM node:18-alpine AS prod
# WORKDIR /app
# RUN npm install -g http-server
# COPY --from=build /app/build ./build
# # Do NOT copy certs here!
# EXPOSE 3000
# CMD ["http-server", "build", "-p", "3000", "-S", "-C", "/etc/ssl/certs/origin.pem", "-K", "/etc/ssl/private/origin.key"]
#
#
# Stage 1: Build
FROM node:20-alpine AS builder

# declare build time variable for Docker
ARG REACT_APP_API_URL

# declare environment variable for React
ENV REACT_APP_API_URL=$REACT_APP_API_URL

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci 
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.27-alpine-slim AS prod
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
# CMD ["nginx", "-g", "daemon off;"]