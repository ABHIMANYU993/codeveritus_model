# FROM node:18-alpine
#
# WORKDIR /usr/src/app
#
# # Install build dependencies for native modules
# RUN apk add --no-cache --virtual .gyp python3 make g++
#
# COPY package.json package-lock.json ./
# RUN npm ci
# COPY . .
# EXPOSE 4000
# CMD ["node", "app.js"]

FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
EXPOSE 4000
CMD ["node", "app.js"]