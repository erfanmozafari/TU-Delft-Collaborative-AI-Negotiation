import logging
import numpy as np
import math
from random import randint
from typing import cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.ValueSet import ValueSet
from decimal import Decimal
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace import LinearAdditive, ValueSetUtilities
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profileconnection import ProfileInterface
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressRounds import ProgressRounds

from agents.time_dependent_agent.time_dependent_agent import TimeDependentAgent


class TemplateAgent(DefaultParty):
    """
    Template agent that offers random bids until a bid with sufficient utility is offered.
    """

    def __init__(self):
        super().__init__()
        self.getReporter().log(logging.INFO, "party is initialized")
        self._profile: ProfileInterface = None
        self._last_received_bid: Bid = None
        self.bidList: list[Bid] = []
        self.bidListOpp: list[Bid] = []
        self.weightList: list[Decimal] = []
        self.weightListLastBidOpp: list[Decimal] = []
        self.weightListOpp: list[Decimal] = []
        self.deltas: list[Decimal] = []
        self.taus: list[Decimal] = []
        self.tau_gen: Decimal = Decimal(0.25)

    def notifyChange(self, info: Inform):
        """This is the entry point of all interaction with your agent after is has been initialised.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(info, Settings):
            self._settings: Settings = cast(Settings, info)
            self._me = self._settings.getID()


            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self._progress: ProgressRounds = self._settings.getProgress()

            # the profile contains the preferences of the agent over the domain
            self._profile = ProfileConnectionFactory.create(
                info.getProfile().getURI(), self.getReporter()
            )

            profile: LinearAdditive = self._profile.getProfile()
            self.weightList: list[Decimal] = list(profile.getWeights().values())
            self.weightListOpp = np.full(len(self.weightList), Decimal(round(1 / len(self.weightList))))
            self.deltas = self._deltas()
            self.taus = self._taus()
        # ActionDone is an action send by an opponent (an offer or an accept)
        elif isinstance(info, ActionDone):
            action: Action = cast(ActionDone, info).getAction()

            # if it is an offer, set the last received bid
            if isinstance(action, Offer):
                self._last_received_bid = cast(Offer, action).getBid()
                self.bidListOpp.append(self._last_received_bid)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(info, YourTurn):
            # execute a turn
            action = self._myTurn()
            if action is Offer:
                self.bidList.append(action.getBid())

            # log that we advanced a turn
            self._progress = self._progress.advance()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(info, Finished):
            # terminate the agent MUST BE CALLED
            self.terminate()
        else:
            self.getReporter().log(
                logging.WARNING, "Ignoring unknown info " + str(info)
            )

    # lets the geniusweb system know what settings this agent can handle
    # leave it as it is for this course
    def getCapabilities(self) -> Capabilities:
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    # terminates the agent and its connections
    # leave it as it is for this course
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        super().terminate()
        if self._profile is not None:
            self._profile.close()
            self._profile = None

    #######################################################################################
    ########## THE METHODS BELOW THIS COMMENT ARE OF MAIN INTEREST TO THE COURSE ##########
    #######################################################################################

    # give a description of your agent
    # Overrride
    def getDescription(self) -> str:
        return "Template agent for Collaborative AI course"

    # execute a turn
    # Override
    def _myTurn(self):
        # check if the last received offer if the opponent is good enough
        if self._isGood(self._last_received_bid):
            # if so, accept the offer
            action = Accept(self._me, self._last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self._findBid()
            action = Offer(self._me, bid)

        # send the action
        self.getConnection().send(action)
        return action



    def _deltas(self) -> list[Decimal]:
        deltas: list[Decimal] = []
        for i in range(len(self.weightList)):
            delta: Decimal = (self.weightListOpp[i] - self.weightList[i])/(self.weightListOpp[i] + self.weightList[i])
            deltas.append(delta)
        return deltas

    def _taus(self) -> list[Decimal]:
        taus: list[Decimal] = []
        for i in range(len(self.weightList)):
            tau: Decimal = self.tau_gen * (1 + self.deltas[i])
            taus.append(tau)
        return taus

    # Consider a switch in decision making as half-hearted bidding and therefore reduce
    # the wight associated with that issue: (https://www.youtube.com/watch?v=xNONxu06XWA).
    # Similiarly, consider persistence on issue values
    # as an assertive stance and therefore put more weight on it.
    # A switch in the initial rounds translates to less reduction than in later rounds : w - w ** (n-i+1)
    def _estimateOppWeights(self, bids: list[Bid]) -> list[Decimal]:
        n_issues = len(bids[0].getIssues())
        ws = [1.0/n_issues] * len(n_issues)
        i=0
        n = len(bids)
        while (i < n-1):
            B1 = bids[i]
            B2 = bids[i+1]
            δs = [(1 if B1[i] == B2[i] else -1) for i in range(len(B1))]
            ws = ws + ws * [abs(x)**(n-i) for x in δs/n]
            normalized_ws = ws / np.sqrt(np.sum(ws**2))
            i += 1
        return normalized_ws

    def _estimateOppWeights_2(self, current_estimated_opp_weights, previous_estimated_opp_weights) -> list[Decimal]:
        n = len(self.weightList)
        r = np.zeros(n)
        keys: list[int] = np.arange(n).tolist()

        # Give each attribute an id and sort the attributes based on their weight
        dict_1: dict[int, Decimal] = dict(zip(keys, current_estimated_opp_weights))
        sorted_dict_1: dict[int, Decimal] = dict(sorted(dict_1.items(), key=lambda item: item[1]))
        dict_2: dict[int, Decimal] = dict(zip(keys, previous_estimated_opp_weights))
        sorted_dict_2: dict[int, Decimal] = dict(sorted(dict_2.items(), key=lambda item: item[1]))

        # R value depending on the distance in ordering
        R = [6, 4, 3, 1, 0.5]

        # Attributes ids ordered on their weights
        sorted_keys_1 = sorted_dict_1.keys()
        sorted_keys_2 = sorted_dict_2.keys()

        for i in range(n):
            for j in range(n):
                # See how much the attribute differs from the previous weights. Based on a 5 point scale
                if sorted_keys_2[j] == sorted_keys_1[i]:
                    r[sorted_keys_1[i]] = R[math.floor(abs(i-j) * 5/n)]

        sum_r = sum(r)
        # Calculate the new weights based on the R values
        new_weights: list[Decimal] = []
        for i in range(n):
            new_weights.append(Decimal(r[i] / sum_r))

        return new_weights

    def _estimateOppWeights_3(self) -> list[Decimal]:
        if len(self.bidListOpp) >= 6:
            current_estimated_opp_weights = self._estimateOppWeights(self.bidListOpp)
            new_weights = self._estimateOppWeights_2(self.weightListOpp, current_estimated_opp_weights)
            self.weightListOpp = new_weights

        return self.weightListOpp


        #Todo combine the two function together with the found tau values


    def _analyzeOpponent(self): pass
        # Todo Analyze the weights?


    def _analyzeOppenentStrategy(self): pass
        # Todo Analyze the bidding strategy
        # Todo Analyze the acceptance strategy maybe??

    # method that checks if we would agree with an offer
    # Override
    def _isGood(self, bid: Bid) -> bool:
        if bid is None:
            return False
        profile = self._profile.getProfile()

        progress = self._progress.get(0)
        if isinstance(profile, UtilitySpace):

            reservation_bid = profile.getReservationBid()
            if reservation_bid is None:
                return False
            reservation_value = profile.getUtility(reservation_bid)
            # print(reservation_value)
            utility_target = (reservation_value + 1) / 2
            boulware_e = .2
            ft1 = round(Decimal(float(1 - pow(progress, 1 / boulware_e))), 6)  # defaults ROUND_HALF_UP
            final = max(min((reservation_value + (utility_target - reservation_value) * ft1), utility_target), reservation_value)
            # print(final)
            # final = reservation_value + (utility_target - reservation_value) * ft1

            return profile.getUtility(bid) > final



        # progressArray = [0, 0.2, 0.4, 0.6, 0.8]
        # utilityArray = [0, 0.9, 0.8, 0.7, 0.6]
        # for i in range(1, len(progressArray)):
        #     if progressArray[i - 1] <= progress <= progressArray[i]:
        #         return profile.getUtility(bid) > utilityArray[i]

        # very basic approach that accepts if the offer is valued above 0.6 and
        # 80% of the rounds towards the deadline have passed
        # return profile.getUtility(bid) > 0.6 and progress > 0.8

    def _isGoodOpp(self, bid: Bid) -> bool:
        opp_utility = []
        profile = self._profile.getProfile()

        print(profile)
        if isinstance(profile, UtilitySpace):
            utilities = profile.getUtilities()
            keys: list[str] = list(utilities.keys())
            for i in range(len(keys)):
                utility = utilities[keys[i]]
                value = utility.getUtility(bid.getValue(keys[i]))
                opp_utility.append(value / self.weightList[i] * self.weightListOpp[i])

        opp_utility = np.array(opp_utility)
        print(opp_utility)
        normalized = opp_utility / np.sqrt(np.sum(opp_utility**2))
        return np.sum(normalized) > 0.2


        # Todo implemtent is good function based on the opponents weights


    # Override
    def _findBid(self) -> Bid:
        # compose a list of all possible bids
        domain = self._profile.getProfile().getDomain()
        all_bids = AllBidsList(domain)

        # take 50 attempts at finding a random bid that is acceptable to us
        for _ in range(200):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            if self._isGood(bid) and self._isGoodOpp(bid):
                print(bid)
                break
        return bid
