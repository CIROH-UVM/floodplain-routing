DROP TABLE IF EXISTS merged;
CREATE TABLE merged (NHDPlusID REAL, ReachCode REAL);

WITH RECURSIVE RecursiveJoin AS (
    SELECT
        nhdplusid,
        tonode,
        fromnode,
        totdasqkm,
		reachcode
    FROM
        NHDPlusFlowlineVAA
    WHERE
        mainstem = 1

    UNION

    SELECT
        us.nhdplusid,
        us.tonode,
        us.fromnode,
        us.totdasqkm,
		ds.reachcode
    FROM
        NHDPlusFlowlineVAA us
    JOIN
        RecursiveJoin ds
    ON
        us.tonode = ds.fromnode
	WHERE
		us.mainstem != 1
)

INSERT INTO merged (NHDPlusID, ReachCode)
SELECT nhdplusid, reachcode 
FROM RecursiveJoin;

DELETE FROM merged WHERE rowid NOT IN (SELECT MIN(rowid) FROM  merged GROUP BY NHDPlusID);