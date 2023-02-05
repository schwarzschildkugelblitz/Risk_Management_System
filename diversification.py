import numpy as np
import openai



class DiversificationMeasure:
    """
    A class to measure sectoral diversification and market capitalization size diversification
    using the entropy measure
    """
    def __init__(self, weights, sector_weights, cap_weights):
        """
        Initializes the class with the portfolio weights, sector weights, and market capitalization
        weights for each stock in the portfolio

        :param weights: list of float, the portfolio weights for each stock
        :param sector_weights: list of float, the sector weights for each stock
        :param cap_weights: list of float, the market capitalization weights for each stock
        """
        self.weights = np.array(weights)
        self.sector_weights = np.array(sector_weights)
        self.cap_weights = np.array(cap_weights)

    def entropy(self, weights):
        """
        Calculates the entropy of a set of weights

        :param weights: list of float, the weights to calculate the entropy of
        :return: float, the entropy of the weights
        """
        return -np.sum(weights * np.log2(weights))

    def sectoral_diversification(self):
        """
        Measures sectoral diversification by calculating the entropy of the
        sector weights of a portfolio

        :return: float, the entropy of the sector weights
        """
        sector_weights = self.sector_weights / np.sum(self.sector_weights)
        return self.entropy(sector_weights)

    def cap_size_diversification(self):
        """
        Measures market capitalization size diversification by calculating the entropy of
        the market cap weights of a portfolio

        :return: float, the entropy of the market cap weights
        """
        cap_weights = self.cap_weights / np.sum(self.cap_weights)
        return self.entropy(cap_weights)

    def diversification_ratio(self): 
        """
        Calculates the diversification ratio of a portfolio

        :return: float, the diversification ratio of a portfolio
        """
        return self.sectoral_diversification() / self.cap_size_diversification()
    
class PortfolioDiversification:
    def __init__(self, sector_weights, cap_weights, sectors, market_caps, stocks):
        """
        Initializes the class with the sector and market capitalization weights for each stock in a portfolio,
        and the sector, market capitalization and name of each stock

        :param sector_weights: list of float, the sector weights for each stock
        :param cap_weights: list of float, the market capitalization weights for each stock
        :param sectors: list of str, the sector of each stock
        :param market_caps: list of float, the market capitalization of each stock
        :param stocks: list of str, the name of each stock
        """
        self.sector_weights = np.array(sector_weights)
        self.cap_weights = np.array(cap_weights)
        self.sectors = sectors
        self.market_caps = market_caps
        self.stocks = stocks

    def diversification_suggestion(self):
        """
        Gives a suggestion for diversification and possible stocks to consider to balance the portfolio,
        based on the sector and market capitalization weights of a portfolio

        :return: str, a suggestion for diversification and possible stocks to consider
        """
        sector_entropy = -np.sum(self.sector_weights * np.log2(self.sector_weights))
        cap_entropy = -np.sum(self.cap_weights * np.log2(self.cap_weights))

        if sector_entropy < 0.5:
            sector_suggestion = "Underweight"
        elif sector_entropy >= 0.5 and sector_entropy <= 1.5:
            sector_suggestion = "Adequately weighted"
        else:
            sector_suggestion = "Overweight"

        if cap_entropy < 0.5:
            cap_suggestion = "Underweight"
        elif cap_entropy >= 0.5 and cap_entropy <= 1.5:
            cap_suggestion = "Adequately weighted"
        else:
            cap_suggestion = "Overweight"

        suggestion = f"Sectoral Diversification: {sector_suggestion}. Market Cap Size Diversification: {cap_suggestion}"

        if sector_suggestion == "Underweight":
            sector_counts = np.unique(self.sectors, return_counts=True)[1]
            max_count = max(sector_counts)
            sectors_to_consider = [sector for i, sector in enumerate(np.unique(self.sectors)) if sector_counts[i] < max_count]
            stocks_to_consider = [self.stocks[i] for i in range(len(self.stocks)) if self.sectors[i] in sectors_to_consider]
            suggestion += f"\n\nConsider adding stocks from the following sectors: {sectors_to_consider}"
            suggestion += f"\n\nSome stocks to consider: {stocks_to_consider}"

        if cap_suggestion == "Underweight":
            market_cap_mean = np.mean(self.market_caps)
            stocks_to_consider = [self.stocks[i] for i in range(len(self.stocks)) if self.market_caps[i] > market_cap_mean]
            suggestion += f"\n\nConsider adding stocks with market capitalization greater than the average market capitalization: {market_cap_mean}"
            suggestion += f"\n\nSome stocks to consider: {stocks_to_consider}"
        return suggestion

    def diversification_suggestion_using_chatgpt(self):
        """
        Gives a suggestion for diversification and possible stocks to consider to balance the portfolio using the gpt3 model,
        based on the sector and market capitalization weights of a portfolio

        :return: str, a suggestion for diversification and possible stocks to consider
        """
        sector_entropy = -np.sum(self.sector_weights * np.log2(self.sector_weights))
        cap_entropy = -np.sum(self.cap_weights * np.log2(self.cap_weights))

        if sector_entropy < 0.5:
            sector_suggestion = "Underweight"
        elif sector_entropy >= 0.5 and sector_entropy <= 1.5:
            sector_suggestion = "Adequately weighted"
        else:
            sector_suggestion = "Overweight"

        if cap_entropy < 0.5:
            cap_suggestion = "Underweight"
        elif cap_entropy >= 0.5 and cap_entropy <= 1.5:
            cap_suggestion = "Adequately weighted"
        else:
            cap_suggestion = "Overweight"

        suggestion = f"Sectoral Diversification: {sector_suggestion}. Market Cap Size Diversification: {cap_suggestion}"

        if sector_suggestion == "Underweight" or cap_suggestion == "Underweight":
            openai.api_key = "API_KEY_HERE" # Replace with your API key

            """
            can make the prompt more specific by adding the sector weights and market cap weights
            Balraj please consider 
            """


            prompt = f"Please suggest some stocks for a portfolio that is {sector_suggestion} in sectoral diversification and {cap_suggestion} in market capitalization size diversification."


            completions = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0
            )
            suggestion += f"\n\n{completions['choices'][0]['text']}"
        return suggestion

