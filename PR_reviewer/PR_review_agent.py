from flask import Flask, request
from llm_utils import LLM_reviewer
from pr_utilities import PullRequest
import os
from cfg import *

SECRET_TOKEN = os.getenv("SECRET_TOKEN")
PERSONAL_AUTHENTICATION_TOKEN = os.getenv("PERSONAL_AUTHENTICATION_TOKEN")
app = Flask(__name__)

@app.before_request
def check_auth():
    token = request.headers.get("Authorization")
    if token != f"Bearer {SECRET_TOKEN}":
        return {"error": "Unauthorized"}, 403

@app.route('/review', methods=['GET'])
def review_pr():
    """Endpoint to trigger PR review by the agent.
    Expects a query parameter 'pr' indicating the PR number to review.

    Returns a JSON response indicating success or failure.
    """
    pr_number = request.args.get('pr')
    pr_number = int(pr_number)
    if not pr_number:
        return {"error": "PR number missing"}, 400
    print(f"Received request to review PR #{pr_number}")
    pr = PullRequest(PERSONAL_AUTHENTICATION_TOKEN, REPOSITORY_PATH, pr_number)
    changes = pr.get_pr_differences()
    last_commit = pr.pull_request.head.sha
    reviewer_agent = LLM_reviewer(LLM_MODEL)
    total_changes = len(changes)
    final_comments = []
    for i, change in enumerate(changes):
        if 'content' in change:
            comments = reviewer_agent.review_document(change['content'], change['patch'], change['file_status'])

            valid_lines = pr.get_valid_lines_from_diff(change['patch'])
            for key, comment in comments.items():
                if key != 'final':
                    requested_line = int(key)
                    valid_line = pr.get_closest_valid_line(valid_lines, requested_line)
                    if valid_line:
                        print(f"Creating inline comment on PR {pr_number}, at line {valid_line}")
                        pr.pull_request.create_review_comment(body=comment, commit=last_commit, path= change['new_file_path'], line = valid_line)
                # Append final comments to summarize all general comments
                else:
                    print(key, comment)
                    final_comments.append(comment)

        # if last file to review ask llm to a summary
        if i == (total_changes-1) and len(final_comments) > 0:
            final_summary = reviewer_agent.generate_general_comment(" ".join(final_comments))
            pr.pull_request.create_issue_comment(final_summary)

    return {"message": f"Agent triggered for PR #{pr_number}"}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)