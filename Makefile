.PHONY: compose-env backend-env check-env docker-setup docker-up docker-smoke docker-build-extension docker-down docker-logs

compose-env:
	@test -f .env.compose || cp .env.compose.example .env.compose
	@echo "Compose env ready: .env.compose"

backend-env:
	@test -f backend/.env || cp backend/.env.example backend/.env
	@echo "Backend env ready: backend/.env"

check-env:
	node scripts/check-env.mjs backend/.env

docker-setup: compose-env backend-env check-env
	@echo "Docker onboarding files are ready."

docker-up: docker-setup
	docker compose --env-file .env.compose up --build -d backend

docker-smoke:
	docker compose --env-file .env.compose --profile check up backend-smoke

docker-build-extension:
	docker compose --env-file .env.compose --profile build run --rm extension-builder

docker-down:
	docker compose --env-file .env.compose down

docker-logs:
	docker compose --env-file .env.compose logs -f backend
