import openai
from colorama import Fore, Style, init
import json
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential

# 初始化colorama和日志
init(autoreset=True)
logging.basicConfig(level=logging.INFO)

class Actor:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    def generate_response(self, prompt, previous_response=None, critic_feedback=None, is_outline=True):
        try:
            full_prompt = f"Original question: {prompt}\n"
            if previous_response:
                full_prompt += f"Previous response: {previous_response}\n"
            if critic_feedback:
                full_prompt += f"Critic's feedback: {critic_feedback}\n"
            
            if is_outline:
                full_prompt += "Please provide a detailed academic paper outline based on the above information. Ensure your outline includes all necessary sections and subsections, and fully utilizes all aspects of the original prompt:"
            else:
                full_prompt += "Based on the provided outline, please generate a complete and detailed academic paper. Ensure each section is fully developed, addresses all points in the outline, and incorporates all aspects of the original prompt. The total word count should be approximately 6000 words, with a balanced distribution across sections. The methodology and experimental sections should be more detailed. If you reach the token limit, please indicate where you had to stop and what topics still need to be covered:"

            logging.info(Fore.YELLOW + "Actor Input: %s", full_prompt)
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a sophisticated language model tasked with generating high-quality, academically rigorous responses. Your output should be either a detailed outline or a complete academic paper, depending on the request. Ensure academic rigor, coherence, and comprehensive coverage of all provided information throughout."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=4000  # Adjust this value as needed
            )
            logging.info(Fore.GREEN + "Actor Output: %s", response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in generate_response: {str(e)}")
            raise

class Critic:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    def evaluate(self, original_prompt, response, iteration, previous_feedback=None, is_outline=True):
        try:
            prompt = f"Original question: {original_prompt}\nResponse to evaluate: {response}\nIteration: {iteration}\n"
            if previous_feedback:
                prompt += f"Previous feedback: {previous_feedback}\n"
            
            if is_outline:
                prompt += "Evaluate this outline. Provide detailed feedback on its structure, completeness, and coherence. Ensure all aspects of the original prompt are addressed. Suggest improvements or additions where necessary:"
            else:
                prompt += "Evaluate this academic paper. Provide detailed feedback on the completeness and depth of each section. Ensure all parts of the paper are fully developed, coherent, and address all aspects of the original prompt. The paper should be approximately 6000 words long with a balanced distribution across sections, and more detailed methodology and experimental sections. Highlight any areas that need expansion or clarification, and note any important topics that may have been omitted due to token limitations:"
            
            logging.info(Fore.YELLOW + "Critic Input: %s", prompt)
            evaluation = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator tasked with critically assessing academic outlines and papers. Your feedback should be comprehensive, specific, and actionable, aimed at ensuring high academic standards and complete coverage of the original prompt."},
                    {"role": "user", "content": prompt}
                ]
            )
            logging.info(Fore.GREEN + "Critic Output: %s", evaluation.choices[0].message.content)
            return evaluation.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in evaluate: {str(e)}")
            raise

def actor_critic_loop(api_key, initial_prompt, iterations=2):
    actor = Actor(api_key)
    critic = Critic(api_key)
    
    results = []
    previous_response = None
    critic_feedback = None

    # First, generate and refine the outline
    for i in range(iterations):
        try:
            response = actor.generate_response(initial_prompt, previous_response, critic_feedback, is_outline=True)
            evaluation = critic.evaluate(initial_prompt, response, i+1, critic_feedback, is_outline=True)
            
            results.append({
                "iteration": i + 1,
                "response": response,
                "evaluation": evaluation,
                "type": "outline"
            })

            previous_response = response
            critic_feedback = evaluation

            if is_outline_complete(response) and all_prompt_aspects_covered(response, initial_prompt):
                break
        except Exception as e:
            logging.error(f"Error in outline iteration {i+1}: {str(e)}")
            break

    # Then, generate the full paper based on the final outline
    if results:
        final_outline = results[-1]["response"]
        try:
            full_paper = generate_full_paper(actor, final_outline, initial_prompt)
            paper_evaluation = critic.evaluate(initial_prompt, full_paper, len(results)+1, None, is_outline=False)
            
            results.append({
                "iteration": len(results) + 1,
                "response": full_paper,
                "evaluation": paper_evaluation,
                "type": "full_paper"
            })
        except Exception as e:
            logging.error(f"Error in generating full paper: {str(e)}")

    return results

def generate_full_paper(actor, outline, initial_prompt):
    sections = parse_outline(outline)
    full_paper = ""
    
    for section, subsections in sections.items():
        section_content = actor.generate_response(f"Generate the {section} section of the paper based on this outline: {subsections}\nEnsure it addresses all aspects of the original prompt: {initial_prompt}", is_outline=False)
        full_paper += f"\n\n{section}\n{section_content}"
    
    return full_paper

def parse_outline(outline):
    sections = {}
    current_section = None
    current_subsection = None
    
    for line in outline.split('\n'):
        line = line.strip()
        if line.startswith('# '):
            current_section = line
            sections[current_section] = {}
        elif line.startswith('## '):
            current_subsection = line
            sections[current_section][current_subsection] = []
        elif current_subsection and line:
            sections[current_section][current_subsection].append(line)
    
    return sections

def single_actor_response(api_key, prompt):
    actor = Actor(api_key)
    try:
        outline = actor.generate_response(prompt, is_outline=True)
        full_paper = generate_full_paper(actor, outline, prompt)
        return outline, full_paper
    except Exception as e:
        logging.error(f"Error in single_actor_response: {str(e)}")
        return None, None

def compare_responses(api_key, initial_prompt):
    try:
        logging.info(Fore.CYAN + "Single Actor Response:")
        single_outline, single_full_paper = single_actor_response(api_key, initial_prompt)
        if single_outline and single_full_paper:
            logging.info(Fore.GREEN + "Outline:\n" + single_outline)
            logging.info(Fore.GREEN + "Full Paper:\n" + single_full_paper)
            logging.info(Fore.RESET + "\n---\n")

        logging.info(Fore.CYAN + "Actor-Critic System Responses:")
        ac_results = actor_critic_loop(api_key, initial_prompt)
        for result in ac_results:
            logging.info(Fore.YELLOW + f"Iteration {result['iteration']} ({result['type']}):")
            logging.info(Fore.GREEN + f"Response: {result['response']}")
            logging.info(Fore.MAGENTA + f"Evaluation: {result['evaluation']}")
            logging.info(Fore.RESET + "---")

        logging.info(Fore.CYAN + "\nComparison:")
        logging.info(Fore.BLUE + f"Single Actor Outline Length: {len(single_outline) if single_outline else 0}")
        logging.info(Fore.BLUE + f"Single Actor Full Paper Length: {len(single_full_paper) if single_full_paper else 0}")
        logging.info(Fore.BLUE + f"Actor-Critic Final Paper Length: {len(ac_results[-1]['response']) if ac_results else 0}")
        logging.info(Fore.BLUE + f"Actor-Critic Iterations: {len(ac_results)}")

        # 保存结果到文件
        save_results(single_outline, single_full_paper, ac_results)

    except Exception as e:
        logging.error(f"Error in compare_responses: {str(e)}")

def save_results(single_outline, single_full_paper, ac_results):
    try:
        # 保存单个Actor的结果
        if single_outline:
            with open('single_actor_outline.txt', 'w', encoding='utf-8') as f:
                f.write(single_outline)
        if single_full_paper:
            with open('single_actor_full_paper.txt', 'w', encoding='utf-8') as f:
                f.write(single_full_paper)

        # 保存Actor-Critic系统的结果
        if ac_results:
            with open('ac_outline.txt', 'w', encoding='utf-8') as f:
                f.write(next(result['response'] for result in ac_results if result['type'] == 'outline'))
            with open('ac_full_paper.txt', 'w', encoding='utf-8') as f:
                f.write(next(result['response'] for result in ac_results if result['type'] == 'full_paper'))

            # 保存所有的Critic过程
            with open('critic_process.json', 'w', encoding='utf-8') as f:
                json.dump(ac_results, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"Error in save_results: {str(e)}")

def is_outline_complete(outline):
    main_sections = ["Abstract", "Introduction", "Related Work", "Methodology", "Experimental Setup", "Results", "Discussion", "Conclusion"]
    return all(section.lower() in outline.lower() for section in main_sections)

def all_prompt_aspects_covered(response, prompt):
    key_aspects = ["Genetic Algorithms", "Large Language Models", "Behavior Trees"]
    return all(aspect.lower() in response.lower() for aspect in key_aspects)

# Example usage:
api_key = "sk-"
initial_prompt = """"""
compare_responses(api_key, initial_prompt)
