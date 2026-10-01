# Сайт документации: сборка из docs/ (Astro + Starlight) и раздача статики через nginx.
#
# Собирать из корня репозитория — сайту нужны docs/ и файлы репозитория для ссылок на код:
#   docker buildx build --platform linux/amd64 -f site/Dockerfile \
#     -t harbor.openidealab.com/llm-arch-research/site:latest .
#
# SITE_URL — адрес сайта (канонические ссылки и sitemap), SITE_BASE — путь от корня домена.

FROM node:24-slim AS build
# git — для дат <lastmod> в карте сайта (scripts/lastmod.mjs); .git попадает в контекст сборки,
# но в образ nginx уходит только dist/
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /repo/site
COPY site/package.json site/package-lock.json ./
RUN npm ci --no-audit --no-fund
# Весь репозиторий (без лишнего — см. Dockerfile.dockerignore): плагин ссылок проверяет,
# файл или папка стоит за ссылкой на код, чтобы вести на blob или tree в GitHub
COPY . /repo
ARG SITE_URL=https://llm-arch-research.openidealab.com
ARG SITE_BASE=/
ENV SITE_URL=${SITE_URL} SITE_BASE=${SITE_BASE}
RUN npm run build

FROM nginx:1.29-alpine
COPY site/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /repo/site/dist /usr/share/nginx/html
EXPOSE 80
