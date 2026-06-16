from pathlib import Path
import yaml

try:
    from .Skill import Skill
except ImportError:
    from Skill import Skill


class SkillRegistry:
    def __init__(self, *skills_folders: Path | str):
        self.skills_folders = [Path(folder) for folder in skills_folders if folder is not None]
        self.skills_folders.append(self._find_project_root() / "skills")
        self.skills_folders = [folder for folder in self.skills_folders if folder.exists()]
        self.name_to_skill: dict[str, Skill] = {}

    def get_skill_content(self, skill: Skill) -> str:
        if not skill.get("content"):
            with open(skill.get("path"), "r", encoding="utf-8") as f:
                skill["content"] = f.read()
        return skill.get("content")

    def get_all_skills(self):
        if not self.name_to_skill:
            self._read_skills_from_folder()
        return iter(self.name_to_skill.values())

    def get_skill_by_name(self, name: str) -> Skill:
        if not self.name_to_skill:
            self._read_skills_from_folder()
        if name not in self.name_to_skill:
            raise ValueError(f"Skill {name} not found")
        return self.name_to_skill[name]

    @staticmethod
    def _find_project_root() -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists():
                return parent
        raise FileNotFoundError("Could not find project root (no .git directory found)")

    def _read_skills_from_folder(self):
        for skills_folder in self.skills_folders:
            if not skills_folder.exists():
                continue
            for skill_folder in skills_folder.iterdir():
                if not skill_folder.is_dir():
                    continue
                skill_md_path = skill_folder / "SKILL.md"
                if not skill_md_path.exists():
                    continue
                metadata = self._read_skill_metadata(skill_md_path)
                skill = Skill(name=metadata["name"], description=metadata["description"], path=skill_md_path)
                self.name_to_skill[skill.get("name")] = skill

    def _read_skill_metadata(self, skill_file: Path | str) -> dict[str, str]:
        skill_file_path = Path(skill_file)

        with open(skill_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f"{skill_file_path} is empty")

        end_front_matter_idx = None
        if lines[0].strip() == "---":
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    end_front_matter_idx = i
                    break

        if end_front_matter_idx is None:
            metadata = {}
            metadata["name"] = skill_file_path.parent.name
            metadata["description"] = self._extract_first_markdown_paragraph(lines)
            return metadata

        yaml_content = "".join(lines[i] for i in range(1, end_front_matter_idx))
        metadata = yaml.safe_load(yaml_content) or {}

        if not isinstance(metadata, dict):
            raise ValueError(f"{skill_file_path} YAML must be a dictionary")

        if "name" not in metadata:
            metadata["name"] = skill_file_path.parent.name

        if "description" not in metadata:
            markdown_lines_iter = (lines[i] for i in range(end_front_matter_idx + 1, len(lines)))
            metadata["description"] = self._extract_first_markdown_paragraph(markdown_lines_iter)

        return metadata

    def _extract_first_markdown_paragraph(self, lines) -> str:
        in_code_block = False
        paragraph_lines: list[str] = []

        for raw_line in lines:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            if not stripped:
                if paragraph_lines:
                    break
                continue

            # Skip headings as they are titles, not paragraph content.
            if stripped.startswith("#") and not paragraph_lines:
                continue

            paragraph_lines.append(stripped)

        return " ".join(paragraph_lines)
