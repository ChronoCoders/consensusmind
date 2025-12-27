pub struct Prompts;

impl Prompts {
    pub fn system_prompt() -> &'static str {
        "You are an expert researcher in blockchain consensus mechanisms. You analyze academic papers, generate hypotheses, and conduct rigorous experiments. Your responses are precise, well-cited, and technically accurate."
    }

    pub fn format_user_query(query: &str) -> String {
        format!("{}\n\n{}", Self::system_prompt(), query)
    }
}
