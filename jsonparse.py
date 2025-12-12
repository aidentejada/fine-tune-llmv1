import json
import os
from datetime import datetime
from typing import List, Dict, Any

class InstagramMessageProcessor:
    def __init__(self, your_name: str, time_gap_hours: float = 1.0, 
                 start_date: str = None):
        """
        initialize the processor.
        
        args:
            your_name: your display name in the instagram data
            time_gap_hours: hours of silence to consider a new conversation
            start_date: only include messages after this date (format: "YYYY-MM-DD")
        """
        self.your_name = your_name
        self.time_gap_ms = time_gap_hours * 3600 * 1000  # convert to milliseconds
        
        # convert start_date to timestamp_ms
        if start_date:
            from datetime import datetime
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            self.start_timestamp_ms = int(dt.timestamp() * 1000)
        else:
            self.start_timestamp_ms = 0
        
    def is_new_conversation(self, prev_msg: Dict, curr_msg: Dict) -> bool:
        """detect if current message starts a new conversation."""
        time_diff = abs(prev_msg['timestamp_ms'] - curr_msg['timestamp_ms'])
        
        # time-based detection
        if time_diff > self.time_gap_ms:
            return True
        
        # topic change indicators (greetings after a gap)
        greetings = ['hey', 'hi', 'yo', 'sup', 'hello', 'ayy', 'aye']
        content = curr_msg.get('content', '').lower().strip()
        
        if time_diff > (self.time_gap_ms / 2):  # half the gap threshold
            if any(content.startswith(word) for word in greetings):
                return True
        
        return False
    
    def clean_message_content(self, msg: Dict) -> str:
        """extract and clean message content."""
        # handle different message types
        if 'content' in msg:
            content = msg['content']
            
            # skip system messages
            skip_phrases = [
                'liked a message',
                'reacted',
                'you sent an attachment',
                'unsent a message'
            ]
            if any(phrase in content.lower() for phrase in skip_phrases):
                return None
            
            return content.strip()
        
        # handle photos
        if 'photos' in msg:
            return "[sent photo]"
        
        # handle shared links
        if 'share' in msg:
            return "[shared link]"
        
        return None
    
    def segment_conversations(self, messages: List[Dict]) -> List[List[Dict]]:
        """break messages into separate conversations."""
        if not messages:
            return []
        
        # filter by date first
        messages = [msg for msg in messages 
                   if msg['timestamp_ms'] >= self.start_timestamp_ms]
        
        if not messages:
            return []
        
        # instagram returns newest first, so reverse
        messages = sorted(messages, key=lambda x: x['timestamp_ms'])
        
        conversations = []
        current_convo = [messages[0]]
        
        for i in range(1, len(messages)):
            if self.is_new_conversation(messages[i-1], messages[i]):
                conversations.append(current_convo)
                current_convo = [messages[i]]
            else:
                current_convo.append(messages[i])
        
        # add the last conversation
        if current_convo:
            conversations.append(current_convo)
        
        return conversations
    
    def format_for_training(self, conversations: List[List[Dict]], 
                           other_person_name: str) -> List[Dict]:
        """
        format conversations for llm training.
        returns list of conversation objects ready for training.
        """
        training_data = []
        
        for conv_idx, conversation in enumerate(conversations):
            formatted_messages = []
            prev_sender = None
            accumulated_content = []
            
            for msg in conversation:
                content = self.clean_message_content(msg)
                if content is None:
                    continue
                
                sender = msg['sender_name']
                
                # accumulate consecutive messages from same sender
                if sender == prev_sender:
                    accumulated_content.append(content)
                else:
                    # save previous accumulated message
                    if accumulated_content and prev_sender:
                        role = "assistant" if prev_sender == self.your_name else "user"
                        combined = "\n".join(accumulated_content)
                        
                        # mark conversation start
                        metadata = {}
                        if len(formatted_messages) == 0:
                            metadata['is_conversation_start'] = True
                        
                        # only mark user messages for generalization (not your messages)
                        if role == "user":
                            combined = f"<TO_GENERALIZE>{combined}</TO_GENERALIZE>"
                        
                        formatted_messages.append({
                            "role": role,
                            "content": combined,
                            "metadata": metadata
                        })
                    
                    # start new accumulation
                    accumulated_content = [content]
                    prev_sender = sender
            
            # don't forget the last accumulated message
            if accumulated_content and prev_sender:
                role = "assistant" if prev_sender == self.your_name else "user"
                combined = "\n".join(accumulated_content)
                
                metadata = {}
                if len(formatted_messages) == 0:
                    metadata['is_conversation_start'] = True
                
                # only mark user messages for generalization (not your messages)
                if role == "user":
                    combined = f"<TO_GENERALIZE>{combined}</TO_GENERALIZE>"
                
                formatted_messages.append({
                    "role": role,
                    "content": combined,
                    "metadata": metadata
                })
            
            # only save conversations with actual exchanges (at least one from each person)
            has_user = any(m['role'] == 'user' for m in formatted_messages)
            has_assistant = any(m['role'] == 'assistant' for m in formatted_messages)
            
            if has_user and has_assistant and len(formatted_messages) >= 2:
                training_data.append({
                    "conversation_id": f"conv_{conv_idx:04d}",
                    "messages": formatted_messages,
                    "message_count": len(formatted_messages)
                })
        
        return training_data
    
    def process_file(self, filepath: str) -> Dict:
        """process a single instagram message json file."""
        # instagram exports use utf-8 with special encoding for emojis
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # fix instagram's double-encoded utf-8 (common issue)
            try:
                content = content.encode('latin1').decode('utf-8')
            except:
                pass  # if it fails, content is already properly encoded
            data = json.loads(content)
        
        # process messages
        messages = data.get('messages', [])
        conversations = self.segment_conversations(messages)
        
        # get other person's name (not needed in output but for processing)
        participants = data.get('participants', [])
        other_person = "Unknown"
        for p in participants:
            if p['name'] != self.your_name:
                other_person = p['name']
                break
        
        training_data = self.format_for_training(conversations, other_person)
        
        return {
            "total_conversations": len(training_data),
            "conversations": training_data
        }
    
    def process_directory(self, directory: str, output_file: str = "training_data.json"):
        """process all message json files in a directory or subdirectories."""
        all_training_data = []
        
        # check if this is the inbox folder with subdirectories
        subdirs = [d for d in os.listdir(directory) 
                  if os.path.isdir(os.path.join(directory, d))]
        
        # if subdirectories exist, process each one
        if subdirs:
            print(f"Found {len(subdirs)} conversation folders\n")
            for subdir in subdirs:
                subdir_path = os.path.join(directory, subdir)
                print(f"Processing folder: {subdir}")
                
                for filename in os.listdir(subdir_path):
                    if filename.startswith('message_') and filename.endswith('.json'):
                        filepath = os.path.join(subdir_path, filename)
                        print(f"  {filename}...")
                        
                        try:
                            result = self.process_file(filepath)
                            if result['total_conversations'] > 0:  # only add if has conversations
                                all_training_data.append(result)
                        except Exception as e:
                            print(f"Error: {e}")
                print()
        else:
            # single directory
            for filename in os.listdir(directory):
                if filename.startswith('message_') and filename.endswith('.json'):
                    filepath = os.path.join(directory, filename)
                    print(f"Processing {filename}...")
                    
                    try:
                        result = self.process_file(filepath)
                        if result['total_conversations'] > 0:
                            all_training_data.append(result)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        
        # save combined output
        output = {
            "metadata": {
                "total_conversations": sum(d['total_conversations'] for d in all_training_data),
                "your_name": self.your_name,
                "date_filter": f"Messages after {self.start_timestamp_ms}" if self.start_timestamp_ms > 0 else "No date filter"
            },
            "conversations": []
        }
        
        # flatten all conversations into single list
        for thread_data in all_training_data:
            output["conversations"].extend(thread_data["conversations"])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessed {len(all_training_data)} conversation threads")
        print(f"Found {output['metadata']['total_conversations']} total conversations")
        print(f"Saved to {output_file}")
        
        return output


# example usage
if __name__ == "__main__":
    # configure these
    YOUR_NAME = "Your Name"  # your exact name from instagram data
    INPUT_DIR = "path/to/inbox"  # path to inbox folder
    OUTPUT_FILE = "training_data.json"
    START_DATE = "2024-08-01"  # only messages after this date (YYYY-MM-DD)
    
    # process
    processor = InstagramMessageProcessor(
        your_name=YOUR_NAME,
        time_gap_hours=1.0,  # adjust this threshold as needed
        start_date=START_DATE
    )
    
    # process single file (for testing)
    # result = processor.process_file(os.path.join(INPUT_DIR, "message_1.json"))
    # print(json.dumps(result, indent=2))
    
    # process entire directory
    processor.process_directory(INPUT_DIR, OUTPUT_FILE)