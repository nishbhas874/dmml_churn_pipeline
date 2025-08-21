"""
DVC Setup Script for Assignment
This script helps set up DVC with GitHub integration for data versioning.
"""

import subprocess
import os
import sys
from pathlib import Path
import json
from typing import Optional

class DVCSetup:
    """DVC setup and configuration helper."""
    
    def __init__(self, project_root: str = ".", github_username: str = None):
        """Initialize DVC setup."""
        self.project_root = Path(project_root)
        self.github_username = github_username
        
    def check_dvc_installed(self) -> bool:
        """Check if DVC is installed."""
        try:
            result = subprocess.run(['dvc', '--version'], 
                                  capture_output=True, text=True, check=False)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_dvc(self):
        """Install DVC if not already installed."""
        if not self.check_dvc_installed():
            print("📦 Installing DVC...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'dvc'], check=True)
                print("✅ DVC installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install DVC: {e}")
                raise
        else:
            print("✅ DVC is already installed")
    
    def initialize_dvc(self):
        """Initialize DVC in the project."""
        print("🔧 Initializing DVC...")
        
        # Initialize DVC
        subprocess.run(['dvc', 'init'], cwd=self.project_root, check=True)
        
        # Create .gitignore entries
        gitignore_path = self.project_root / '.gitignore'
        if not gitignore_path.exists():
            gitignore_path.touch()
        
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        # Add DVC-related entries
        dvc_entries = [
            '# DVC',
            '.dvc/',
            'data/',
            '*.dvc/cache',
            'dvc_remote/'
        ]
        
        for entry in dvc_entries:
            if entry not in content:
                with open(gitignore_path, 'a') as f:
                    f.write(f'\n{entry}')
        
        print("✅ DVC initialized successfully")
    
    def setup_github_integration(self, github_username: str, repo_name: str = None):
        """Setup GitHub integration for DVC."""
        if not github_username:
            print("⚠️  GitHub username not provided, skipping GitHub integration")
            return
        
        print(f"🔗 Setting up GitHub integration for user: {github_username}")
        
        # Create GitHub remote URL
        if repo_name:
            remote_url = f"https://github.com/{github_username}/{repo_name}.git"
        else:
            remote_url = f"https://github.com/{github_username}/dmml_churn_pipeline.git"
        
        try:
            # Add Git remote
            subprocess.run(['git', 'remote', 'add', 'origin', remote_url], 
                         cwd=self.project_root, check=False)
            
            # Setup DVC remote for GitHub
            dvc_remote_url = f"https://github.com/{github_username}/dvc-storage"
            subprocess.run(['dvc', 'remote', 'add', 'origin', dvc_remote_url], 
                         cwd=self.project_root, check=False)
            
            # Set default remote
            subprocess.run(['dvc', 'remote', 'default', 'origin'], 
                         cwd=self.project_root, check=False)
            
            print(f"✅ GitHub integration configured")
            print(f"   Repository: {remote_url}")
            print(f"   DVC Storage: {dvc_remote_url}")
            
        except Exception as e:
            print(f"⚠️  GitHub integration setup failed: {e}")
            print("   You can manually configure GitHub integration later")
    
    def setup_local_remote(self):
        """Setup local remote storage for DVC."""
        print("💾 Setting up local remote storage...")
        
        local_remote_path = self.project_root / 'dvc_remote'
        local_remote_path.mkdir(exist_ok=True)
        
        try:
            # Add local remote
            subprocess.run(['dvc', 'remote', 'add', 'local', str(local_remote_path)], 
                         cwd=self.project_root, check=False)
            
            # Set as default if no other remote is set
            result = subprocess.run(['dvc', 'remote', 'default'], 
                                  cwd=self.project_root, capture_output=True, text=True)
            if 'local' not in result.stdout:
                subprocess.run(['dvc', 'remote', 'default', 'local'], 
                             cwd=self.project_root, check=False)
            
            print(f"✅ Local remote configured: {local_remote_path}")
            
        except Exception as e:
            print(f"⚠️  Local remote setup failed: {e}")
    
    def create_sample_dvc_yaml(self):
        """Create a sample dvc.yaml file for the project."""
        dvc_yaml_content = """# DVC Pipeline Configuration
stages:
  data_ingestion:
    cmd: python src/ingestion/unified_ingestion.py
    deps:
      - src/ingestion/
      - config/config.yaml
    outs:
      - data/raw/huggingface/
      - data/raw/kaggle/
    metrics:
      - metrics/ingestion_metrics.json

  data_validation:
    cmd: python src/validation/data_validator.py
    deps:
      - data/raw/
      - src/validation/
    outs:
      - data/validated/
    metrics:
      - metrics/validation_metrics.json

  data_preprocessing:
    cmd: python src/preprocessing/data_preprocessor.py
    deps:
      - data/validated/
      - src/preprocessing/
    outs:
      - data/processed/
    metrics:
      - metrics/preprocessing_metrics.json

  feature_engineering:
    cmd: python src/transformation/feature_engineer.py
    deps:
      - data/processed/
      - src/transformation/
    outs:
      - data/transformed/
    metrics:
      - metrics/feature_metrics.json

  model_training:
    cmd: python src/models/train_model.py
    deps:
      - data/transformed/
      - src/models/
    outs:
      - models/
    metrics:
      - metrics/model_metrics.json
"""
        
        dvc_yaml_path = self.project_root / 'dvc.yaml'
        with open(dvc_yaml_path, 'w') as f:
            f.write(dvc_yaml_content)
        
        print("✅ Created sample dvc.yaml pipeline configuration")
    
    def create_initial_commit(self):
        """Create initial Git commit with DVC files."""
        try:
            # Add all files
            subprocess.run(['git', 'add', '.'], cwd=self.project_root, check=True)
            
            # Create initial commit
            subprocess.run(['git', 'commit', '-m', 'Initial commit with DVC setup'], 
                         cwd=self.project_root, check=True)
            
            print("✅ Created initial Git commit")
            
        except Exception as e:
            print(f"⚠️  Git commit failed: {e}")
            print("   You can manually commit the changes later")
    
    def setup_complete(self, github_username: str = None, repo_name: str = None):
        """Complete DVC setup process."""
        print("🚀 Starting complete DVC setup...")
        
        # Install DVC
        self.install_dvc()
        
        # Initialize DVC
        self.initialize_dvc()
        
        # Setup local remote
        self.setup_local_remote()
        
        # Setup GitHub integration if username provided
        if github_username:
            self.setup_github_integration(github_username, repo_name)
        
        # Create sample pipeline configuration
        self.create_sample_dvc_yaml()
        
        # Create initial commit
        self.create_initial_commit()
        
        print("\n✅ DVC setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Add your datasets to the data/ directory")
        print("2. Use 'dvc add data/your_dataset.csv' to version your data")
        print("3. Use 'dvc push' to push to remote storage")
        print("4. Use 'dvc pipeline show' to visualize your data pipeline")
        
        if github_username:
            print(f"\n🔗 GitHub Integration:")
            print(f"   - Repository: https://github.com/{github_username}/dmml_churn_pipeline")
            print(f"   - DVC Storage: https://github.com/{github_username}/dvc-storage")

def main():
    """Main function for DVC setup."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup DVC for data versioning')
    parser.add_argument('--github-username', type=str, 
                       help='GitHub username for integration')
    parser.add_argument('--repo-name', type=str, 
                       help='GitHub repository name (optional)')
    parser.add_argument('--project-root', type=str, default='.',
                       help='Project root directory')
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = DVCSetup(project_root=args.project_root, 
                    github_username=args.github_username)
    
    # Run complete setup
    setup.setup_complete(args.github_username, args.repo_name)

if __name__ == "__main__":
    # Interactive setup if no arguments provided
    if len(sys.argv) == 1:
        print("🔧 Interactive DVC Setup")
        print("=" * 40)
        
        project_root = input("Project root directory (default: .): ").strip() or "."
        github_username = input("GitHub username (optional): ").strip() or None
        repo_name = input("Repository name (optional): ").strip() or None
        
        setup = DVCSetup(project_root=project_root, github_username=github_username)
        setup.setup_complete(github_username, repo_name)
    else:
        main()
