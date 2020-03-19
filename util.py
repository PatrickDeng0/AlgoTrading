import os
import pandas as pd
import numpy as np
import datetime as dt


##################################################################
# Define the two types of order and simulate
# Simulate the generate of orders through average trading price of Next tick
# Attention: when the trade-through happens such simulator is unaware

# A single sect could be divided into two parts:
# The active price(orders already on the panel, we could make the deal immediately)
# The passive price(we wait for others eating our ordered in the next tick)

class SellOrder:
    def __init__(self, volume, price, Sect, pre_order=None):
        self.totalVol = volume
        self.price = price
        self.unfill = volume
        self.fill = 0
        self.turnover = 0

        # Market Order have 0 rank
        if isinstance(self.price, str):
            self.rank = 0
        else:
            # Not market order, then we compute the rank
            newrank = 0
            for i in range(10, 20):
                if self.price == Sect[i]:
                    newrank = (Sect[i + 20] // 100) * 100
                    break
            if pre_order is not None and pre_order.price == self.price:
                self.rank = min(newrank, pre_order.rank)
            else:
                self.rank = newrank
        self.termin = False

    def updateTermin(self):
        self.termin = (self.unfill == 0)

    # Judge the trade inter-tick and update the state of Order
    def SimTrade(self, Sects):
        # Copy the origin Sect in case we cancel the origin data
        copy_Sects = np.array(Sects)
        for Sect in copy_Sects:
            # Early break if whole order is executed
            if self.termin:
                break
            # Refresh our rank (if there are cancelled orders, we would move forward)
            if not self.rank == 0:
                newRank = 0
                for i in range(30, 40):
                    if self.price == Sect[i]:
                        newRank = (Sect[i] // 100) * 100
                        break
                self.rank = min(self.rank, newRank)

            if not isinstance(self.price, str):
                # For the prices at active positive, we make the deal immediately
                for i in range(10):
                    price = Sect[i]
                    if price >= self.price and not self.unfill == 0:
                        amount = min(Sect[i + 20], self.unfill)
                        self.turnover += price * amount
                        self.fill += amount
                        self.unfill = self.totalVol - self.fill
                    else:
                        break

                # Matching the active orders, if still remain, we match the passive order
                if Sect[41] > 0 and not self.unfill == 0:
                    price = Sect[10]
                    if price >= self.price:
                        # Rank decreases, and only when rank == 0, we get the order filled
                        amount = max(0, Sect[41] - self.rank)
                        self.rank = max(0, self.rank - Sect[41])
                        exe_price = self.price

                        # No more than order.unfill
                        amount = min(amount, self.unfill)
                        self.turnover += exe_price * amount
                        self.fill += amount
                        self.unfill = self.totalVol - self.fill
            else:
                # Market Order Simulation (matching with bids)
                for i in range(10):
                    price = Sect[i]
                    if self.unfill > 0:
                        amount = min(Sect[i + 20], self.unfill)
                        self.turnover += price * amount
                        self.fill += amount
                        self.unfill = self.totalVol - self.fill
                    else:
                        break
            self.updateTermin()
        # Considering MO failure (when limit-down, we force it to finish it all at ask1 at the final Sect)
        if isinstance(self.price, str) and self.unfill > 0:
            amount = self.unfill
            price = copy_Sects[-1][10]
            self.turnover += price * amount
            self.fill += amount
            self.unfill = self.totalVol - self.fill
            self.updateTermin()
