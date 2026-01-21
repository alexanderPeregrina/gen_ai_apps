from github import Github
import re

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
    
    def get_valid_lines_from_diff(self, patch):
        """
        Parse a unified diff string and return a list of valid line numbers
        (from the new file side) where review comments can be placed.

        Uses new_start and new_count from hunk headers.
        """
        valid_lines = []

        # Regex to capture hunk headers: @@ -old_start,old_count +new_start,new_count @@
        hunk_re = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

        for line in patch.splitlines():
            hunk_match = hunk_re.match(line)
            if hunk_match:
                new_start = int(hunk_match.group(1))
                new_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1

                # Add the range of new lines covered by this hunk
                valid_lines.extend(range(new_start, new_start + new_count))

        return valid_lines


    def get_closest_valid_line(self, valid_lines, requested_line):
        """
        verifies if the requested line is in the valid lines from diff
    
        :param valid_lines: List of valid lines
        :param requested_line: Line extracted from LLM comment
        :return: requested_line if valid, else closest valid line
        """
        if not valid_lines:
            return None

        # Clamp requested line to nearest valid line
        closest_line = min(valid_lines, key=lambda x: abs(x - requested_line))
        return closest_line
