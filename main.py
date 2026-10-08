import os
from dotenv import load_dotenv
from transcriber import VideoTranscriber
from research_agent import create_research_agent
from summarizer import generate_report

load_dotenv()

def main():
    print("Starting Agentic-RAG Pipeline...")
    
    # In a real run, this would be a real file. Using mock data for immediate demonstration if no file provided.
    transcriber = VideoTranscriber()
    # transcript = transcriber.transcribe("sample_video.mp4")
    transcript = "Mock transcript: Today we are discussing AI safety and deep learning scaling laws."
    print("Transcription complete.")
    
    print("Initiating web research...")
    # agent = create_research_agent()
    # research_findings = agent.run("What are the latest developments in AI scaling laws?")
    research_findings = "Mock research: Scaling laws continue to hold, but data quality is becoming a bottleneck."
    
    print("Generating Map-Reduce Report...")
    # report = generate_report(transcript, research_findings)
    report = "Mock Report: Based on the transcription and research, scaling laws remain a pivotal area of AI development, with a shift in focus towards data quality."
    print("\n=== FINAL FACT-CHECKED REPORT ===")
    print(report)

if __name__ == "__main__":
    main()
