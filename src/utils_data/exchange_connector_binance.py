import random
import time
from enum import Enum

import requests
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


class IntervalLetter(Enum):
    SECOND = 's'
    MINUTE = 'm'
    HOUR = 'h'
    DAY = 'd'
    WEEK = 'w'
    MONTH = 'M'


class BinanceConnector:
    BASE_URLS = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
        "https://api4.binance.com",
    ]
    DATA_BASE_URL = "https://data-api.binance.vision"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': self.api_key})
        self.base_url = self.BASE_URLS[0]
        self.weight_used = 0
        self.order_count = 0

    def _get(self, endpoint: str, params: Dict[str, Any], use_data_api: bool = False) -> Dict:
        url = f'{self.DATA_BASE_URL if use_data_api else self.base_url}{endpoint}'
        for attempt in range(5):
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()

                # Update weight and order count
                self.weight_used = int(response.headers.get('X-MBX-USED-WEIGHT-1M', 0))
                self.order_count = int(response.headers.get('X-MBX-ORDER-COUNT-1M', 0))

                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == 4 or response.status_code in [418, 429]:
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 30))
                        print(f'Rate limit exceeded. Waiting for {retry_after} seconds')
                        time.sleep(retry_after)
                    elif response.status_code == 418:
                        print(f'IP has been auto-banned. Please wait and try again later.')
                        raise
                    else:
                        raise
                else:
                    # If it's a 5XX error, try a different base URL
                    if 500 <= response.status_code < 600:
                        self.base_url = random.choice(self.BASE_URLS)
                        print(f'Switching to {self.base_url}')
                    time.sleep(2 ** attempt)  # Exponential backoff

    def get_klines(self, symbol: str, interval_num: int, interval_letter: IntervalLetter,
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None, time_zone: str = '0', limit: int = 500) -> List[List[Any]]:
        """
        Obtiene datos de velas (klines) para un símbolo y intervalo específicos.

        :param symbol: El par de trading (por ejemplo, 'BTCUSDT')
        :param interval_num: El número de intervalos
        :param interval_letter: La letra del intervalo (usar la enumeración IntervalLetter)
        :param start_time: Timestamp en milisegundos para obtener velas desde (inclusive)
        :param end_time: Timestamp en milisegundos para obtener velas hasta (inclusive)
        :param time_zone: Zona horaria para interpretar los intervalos de las velas (por defecto "0" UTC)
        :param limit: Límite de resultados (por defecto 500, máximo 1000)
        :return: Lista de velas
        """
        endpoint = '/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': f'{interval_num}{interval_letter.value}',
            'timeZone': time_zone,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._get(endpoint, params, use_data_api=False)

    def get_exchange_info(self) -> Dict[str, Any]:
        endpoint = '/api/v3/exchangeInfo'
        return self._get(endpoint, {}, use_data_api=False)

    def get_rate_limits(self) -> Dict[str, Any]:
        exchange_info = self.get_exchange_info()
        return {
            'rate_limits': exchange_info.get('rateLimits', []),
            'weight_used': self.weight_used,
            'order_count': self.order_count
        }
