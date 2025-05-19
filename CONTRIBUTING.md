# Contributing to Portfolio News Summarizer

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. **Fork the repository**: Start by forking the repository to your GitHub account.

2. **Clone the repository**: Clone your fork to your local machine.
   ```bash
   git clone https://github.com/yourusername/portfolio-news-summarizer.git
   cd portfolio-news-summarizer
   ```

3. **Create a branch**: Create a new branch for your changes.
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**: Implement your changes, following the code style of the project.

5. **Test your changes**: Make sure your changes don't break any existing functionality.

6. **Commit your changes**: Use clear commit messages.
   ```bash
   git commit -m "Add feature: your feature description"
   ```

7. **Push to your fork**: Push your changes to your GitHub fork.
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a pull request**: Open a pull request from your branch to the main repository.

## Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run tests:
   ```bash
   # Add your tests here
   ```

## Project Structure

- `scraper.py` - Fetches news articles
- `cleaner.py` - Preprocesses articles
- `baseline.py` - Baseline summarization
- `prepare_training_data.py` - Prepares data for fine-tuning
- `finetune.py` - Fine-tunes models
- `inference.py` - Runs inference
- `metrics.py` - Evaluates summaries
- `config.py` - Configuration settings
- `ui/app.py` - Streamlit UI

## Areas for Contribution

- **Additional data sources**: Add more sources for news articles
- **Improved preprocessing**: Enhance text cleaning and preprocessing
- **Model improvements**: Try different architectures or fine-tuning techniques
- **UI enhancements**: Improve the Streamlit interface
- **Documentation**: Improve or expand documentation
- **Tests**: Add more tests for different components

## Style Guidelines

- Follow PEP 8 for Python code
- Include docstrings for all functions, classes, and modules
- Use type hints where appropriate
- Keep functions focused on a single responsibility

## Licensing

By contributing to this project, you agree that your contributions will be licensed under the project's MIT license. 