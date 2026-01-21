# Pull Request Reviewer Agent

This project implements an **AI-powered Pull Request Reviewer** using **Flask**, **GitHub Actions**, and **Ollama models**.  
The agent automatically analyzes code changes and posts review comments on GitHub PRs when triggered.

---

## PR Reviewed Examples

An example of a Pull Request reviewed by this agent can be found here:  
[PR #1 in gen_ai_apps](https://github.com/alexanderPeregrina/gen_ai_apps/pull/1)

This demonstrates how the agent posts inline and summary comments automatically when triggered.

---

## Features
- **Flask App**: Provides a lightweight HTTP server that exposes the review endpoint (`/review`).
- **Automatic PR Comments**: The agent inspects diffs and posts inline or summary comments back to the PR.
- **GitHub Action Integration**: The workflow listens for PR events and triggers the agent when the `review-agent` label is added.
- **Ollama Model Support**: Uses the `qwen3-coder:480b-cloud` model by default, but can be swapped for any local Ollama model.
- **Customizable**: Can be ported to any repository by adjusting the workflow YAML and endpoint URL.

---

## Installation

Refer to the [main repository README](https://github.com/alexanderPeregrina/gen_ai_apps) for instructions on setting up the Python environment.  
This includes creating a virtual environment and installing dependencies.

### Install Ollama
The reviewer agent relies on Ollama to run LLMs locally or in the cloud.

1. Install Ollama from [https://ollama.com](https://ollama.com).
2. Pull the default model:
   ```bash
   ollama pull qwen3-coder:480b-cloud
   ```
3. If you want to use another model, pull it with Ollama and update the configuration

   ```bash
   ollama pull <model-name>
   ```
4. Update the LLM_MODEL variable in cfg.py to point to your chosen model:
   ```bash
    LLM_MODEL = "qwen3-coder:480b-cloud"  # change to your preferred model
    ```
### Running the Agent

To run the agent locally:
```bash
python PR_review_agent.py
```
This starts the Flask server and exposes the /review endpoint.

### Triggering ReviewsThe agent is triggered via a GET request to /review.
GitHub Actions automatically sends this request when the review-agent label is added to a Pull Request.

Example:- Add the label review-agent to a PR.
- GitHub Action calls your Flask app’s /review?pr=<PR_NUMBER> endpoint.
- The agent analyzes the PR and posts comments.

 ## Porting to Your Own Repository
 
 To use this agent in your own repo:
 
1. Copy the workflow YAML file from [here](https://github.com/alexanderPeregrina/gen_ai_apps/blob/main/.github/workflows/review_agent.yaml).

2. Place it under the same path in your repo:
```bash
.github/workflows/review-agent.yml
```

3. Update the workflow to point to your agent’s endpoint URL.
--------

## Exposing the Agent

During development and testing, [ngrok](https://ngrok.com/)  was used to expose the Flask app to the internet.
This allowed GitHub Actions to reach the local server.

Other possible deployment options:-
- Heroku: Deploy the Flask app as a web service.
- AWS Lambda + API Gateway: Run the agent serverless.
- Azure App Service or Google Cloud Run: Containerize and deploy the app.
- Docker + VPS: Host the agent on your own server.
### Workflow Integration
The GitHub Action workflow is responsible for:
- Listening for PR events.
- Checking for the review-agent label.
- Sending a GET request to your Flask app’s /review endpoint.

Make sure to:
- Copy the workflow YAML into .github/workflows in your repo.
- Update the endpoint URL to match your deployment (ngrok, cloud, or server).
