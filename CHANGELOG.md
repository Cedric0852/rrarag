# Changelog

## [v1.2] - 2024-10-22

### Added ✨
- Feedback collection system with thumbs up/thumbs down buttons
- Vector storage for feedback using Qdrant
- Advanced retrieval system with Cohere reranking
- Multi-language example questions interface
- Improved logging system with detailed error tracking

### Changed 🔄
- Enhanced LLM initialization with fallback to mixtral-8x7b-32768
- Updated prompt template with structured response format
- Improved answer processing with cleaner formatting
- Modified chat interface to use message-style display
- Enhanced translation service integration

### Fixed 🐛
- Response formatting and cleaning
- Error handling in feedback submission
- Chat history management
- Vector storage initialization checks
- Translation service reliability

### Technical Updates 🛠
- Added FastEmbedEmbeddings with nomic-embed-text-v1.5-Q model
- Implemented asyncio for better performance
- Added proper vector size verification (768) for feedback collection
- Enhanced error logging with detailed formatting
- Improved service initialization with proper error handling

## [v1.1] - 2024-10-22

### Added ✨
- Gradio interface implementation
- Basic translation service
- Structured logging system
- Privacy notice and disclaimer
- Basic chat functionality

### Changed 🔄
- Initial UI setup with Gradio
- Basic language selection
- Simple question-answer flow
- Basic error handling
- Basic chat history management

### Technical Implementation 🛠
- Initial Groq LLM integration
- Basic Qdrant setup
- Simple prompt template
- Basic async support
- Environment variable configuration

## [v1.0] - 2024-10-22
- Initial release with basic functionality
- Basic question-answering capability
- Simple language support
- Basic error handling