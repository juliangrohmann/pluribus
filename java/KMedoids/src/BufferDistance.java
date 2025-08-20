import elki.database.ids.DBIDRange;
import elki.distance.AbstractDBIDRangeDistance;

import java.io.IOException;

public class BufferDistance extends AbstractDBIDRangeDistance {
	private final RowStripeMapper map;
	
	public BufferDistance(RowStripeMapper map) {
		this.map = map;
	}
	
	@Override
	public double distance(int i, int j) {
		try {
			return map.get(i, j);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public void checkRange(DBIDRange dbidRange) {
	
	}
}
