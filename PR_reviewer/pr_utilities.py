from github import Github

class PullRequest :

    def __init__(self, pat, repo_path, pr_number):
        self.github_account = Github(pat)
        self.repo = self.github_account.get_repo(repo_path)
        self.pull_request = self.repo.get_pull(pr_number)

    def get_pr_differences(self):

        base_commit = self.pull_request.base.sha
        head_commit = self.pull_request.head.sha
        comparison = self.repo.compare(base_commit, head_commit)
        changes = []
        print("Getting differences...")
        for file in comparison.files:
            diffs = {}
            diffs['new_file_path'] = file.filename
            diffs['old_file_path'] = file.previous_filename
            diffs['file_status'] = file.status # added/modified/removed/renamed
            diffs['patch'] = file.patch # unified diff format (str)
            if file.status != 'removed': # only fetch if file still exist
                contents = self.repo.get_contents(file.filename, ref=head_commit)
                diffs['content'] = contents.decoded_content.decode("utf-8")
            changes.append(diffs)
        return changes
